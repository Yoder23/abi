"""Execute the frozen R59 endpoint on the single prospective R60 matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.capability_compiler_phase2_common import sha256_file
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import load_layercake_core
from abi.layercake_host import _truncate_novel_lexical_repetition
from experiments.deep_sparse_adapters_r55 import screen_v1 as r55
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    write_json_once,
    write_jsonl_once,
)
from experiments.irreversible_collapse_invariant_r59 import screen_v1 as r59
from experiments.isolated_capability_cakes_r53 import fit_router


CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
CATALOG_LOCK_SHA256 = "d4ec713575f86a6fd078e0b54528f8e66fe6b464c8ce574868205a05b37496f6"
SOURCE_LOCK_SHA256 = "29993ae1064b6582aa334d57e95564b1f5c69437104236491d206c218e689adb"
SOURCE_RECEIPT_SHA256 = "fc2384e461d5a43ba05e5abe7bcdb47220ab1acec1de990fb100d36e39505f03"
SOURCE_OUTPUTS_SHA256 = "35a5a6e1035035db1663d793257274e3be2b766aab79a00a0476478d0428b596"
EXPECTED_ROWS = 1_400


class ScreenError(RuntimeError):
    pass


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise ScreenError(f"required raw rows absent: {path}")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ScreenError(f"raw rows unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise ScreenError(f"raw rows contain a non-object: {path}")
    return rows


@torch.inference_mode()
def _generate_parent_raw(
    model: Any,
    tokenizer: Any,
    prompt: str,
    route: int,
    maximum: int,
    device: torch.device,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + maximum > model.config.max_tokens:
        raise ScreenError("R60 prompt exceeds candidate context")
    route_tensor = torch.tensor([route], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
        task_routes=route_tensor,
        use_cache=True,
    )
    physical = [tuple(model.last_cake_calls)]
    raw_ids: list[int] = []
    state = {
        "past_key_values": result["past_key_values"],
        "next_logits": result["logits"][:, -1],
    }
    language_model_eos = False
    for _ in range(maximum):
        selected = state["next_logits"][0].argmax(dim=-1).reshape(1)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            language_model_eos = True
            break
        raw_ids.append(token)
        result = model(
            selected[:, None],
            task_routes=route_tensor,
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        physical.append(tuple(model.last_cake_calls))
        if tuple(model.last_cake_calls) != (route,):
            raise ScreenError("parent executed another capability route")
        state = {
            "past_key_values": result["past_key_values"],
            "next_logits": result["logits"][:, -1],
        }
    elapsed = time.perf_counter() - started
    raw_output = tokenizer.decode(
        raw_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output = _truncate_novel_lexical_repetition(raw_output, prompt, threshold=1)
    return {
        "output": output,
        "output_token_ids": tokenizer.encode(output),
        "raw_output": raw_output,
        "raw_output_token_ids": raw_ids,
        "latency_seconds": elapsed,
        "physical": physical,
        "language_model_eos": language_model_eos,
        "lexical_postprocessor_changed_output": output != raw_output,
    }


def _candidate_prefix(parent_raw_ids: list[int]) -> tuple[list[int], str | None]:
    accepted: list[int] = []
    for token in parent_raw_ids:
        candidate = accepted + [token]
        if r59._maximum_identical_token_run(candidate) >= r59.MAXIMUM_IDENTICAL_TOKEN_RUN:
            return accepted, "maximum_identical_token_run"
        accepted.append(token)
    return accepted, None


def run(
    candidate: Path,
    router_path: Path,
    catalog_path: Path,
    source_dir: Path,
    catalog_lock: Path,
    source_lock: Path,
    layercake_root: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R60 screen exists: {output}")
    frozen = (
        (candidate / "model.safetensors", r55.CANDIDATE_SHA256),
        (candidate / "metadata.json", r55.METADATA_SHA256),
        (router_path, r55.ROUTER_SHA256),
        (catalog_path, CATALOG_SHA256),
        (catalog_lock, CATALOG_LOCK_SHA256),
        (source_lock, SOURCE_LOCK_SHA256),
        (source_dir / "receipt.json", SOURCE_RECEIPT_SHA256),
        (source_dir / "source_outputs.jsonl", SOURCE_OUTPUTS_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise ScreenError(f"frozen R60 input changed: {path}")
    r55._preflight(candidate)
    if not torch.cuda.is_available():
        raise ScreenError("R60 prospective screen requires CUDA")

    probes = list(load_probe_catalog(catalog_path)["probes"])
    source_rows = _jsonl(source_dir / "source_outputs.jsonl")
    source = {str(row["probe_id"]): row for row in source_rows}
    counts = Counter(str(row["capability"]) for row in probes)
    if (
        len(probes) != EXPECTED_ROWS
        or len(source) != EXPECTED_ROWS
        or len(counts) != 14
        or set(counts.values()) != {100}
        or set(source) != {str(row["probe_id"]) for row in probes}
    ):
        raise ScreenError("R60 prospective matrix identity changed")

    router = torch.nn.Linear(r49.FEATURES, len(fit_router.CAPABILITY_TO_INDEX))
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(
        candidate, layercake_root=layercake_root, device=device
    )
    model.eval()
    rows = []
    started = time.perf_counter()
    telemetry: dict[str, Any] = {
        "candidate_transitions_checked": 0,
        "accepted_tokens": 0,
        "rejected_boundary_tokens": 0,
        "rejections_by_condition": {
            "maximum_identical_token_run": 0,
            "repeated_novel_lexical_fourgrams": 0,
        },
        "language_model_eos_stops": 0,
        "inherited_r55_lexical_truncations": 0,
    }
    for ordinal, probe in enumerate(probes, 1):
        prompt = str(probe["prompt"])
        expected_route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
        route = int(router(r49._feature(prompt)).argmax())
        parent = _generate_parent_raw(
            model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device
        )
        candidate_output, candidate_tokens, candidate_latency, candidate_physical = r59._generate(
            model,
            tokenizer,
            prompt,
            route,
            int(probe["max_new_tokens"]),
            device,
            telemetry=telemetry,
        )
        candidate_raw_ids, stop_reason = _candidate_prefix(parent["raw_output_token_ids"])
        candidate_raw_output = tokenizer.decode(
            candidate_raw_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        derived_candidate_output = _truncate_novel_lexical_repetition(
            candidate_raw_output, prompt, threshold=1
        )
        if candidate_output != derived_candidate_output:
            raise ScreenError(f"R59 runtime derivation mismatch: {probe['probe_id']}")
        candidate_passed, candidate_score = evaluate_output(
            candidate_output, probe["evaluator"]
        )
        parent_passed, parent_score = evaluate_output(
            parent["output"], probe["evaluator"]
        )
        source_row = source[str(probe["probe_id"])]
        candidate_raw_collapse = _collapse_metrics(
            candidate_raw_ids,
            candidate_raw_output,
            tokenizer.encode(prompt + "\n"),
            prompt,
        )
        candidate_final_collapse = _collapse_metrics(
            candidate_tokens,
            candidate_output,
            tokenizer.encode(prompt + "\n"),
            prompt,
        )
        rows.append(
            {
                "probe_id": probe["probe_id"],
                "capability": probe["capability"],
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "evaluator": probe["evaluator"],
                "route": route,
                "expected_route": expected_route,
                "route_correct": route == expected_route,
                "candidate_output": candidate_output,
                "candidate_output_sha256": hashlib.sha256(
                    candidate_output.encode("utf-8")
                ).hexdigest(),
                "candidate_output_token_ids": candidate_tokens,
                "candidate_raw_output": candidate_raw_output,
                "candidate_raw_output_token_ids": candidate_raw_ids,
                "candidate_runtime_stop_reason": stop_reason,
                "candidate_functional_pass": bool(candidate_passed),
                "candidate_functional_score": float(candidate_score),
                "candidate_raw_collapse": candidate_raw_collapse,
                "candidate_final_collapse": candidate_final_collapse,
                "candidate_latency_seconds": candidate_latency,
                "candidate_maximum_cakes_called": max(map(len, candidate_physical)),
                "candidate_selected_route_only": all(
                    calls == (route,) for calls in candidate_physical
                ),
                "parent_output": parent["output"],
                "parent_output_sha256": hashlib.sha256(
                    parent["output"].encode("utf-8")
                ).hexdigest(),
                "parent_output_token_ids": parent["output_token_ids"],
                "parent_raw_output": parent["raw_output"],
                "parent_raw_output_token_ids": parent["raw_output_token_ids"],
                "parent_functional_pass": bool(parent_passed),
                "parent_functional_score": float(parent_score),
                "parent_latency_seconds": parent["latency_seconds"],
                "parent_lexical_postprocessor_changed_output": parent[
                    "lexical_postprocessor_changed_output"
                ],
                "parent_language_model_eos": parent["language_model_eos"],
                "candidate_parent_exact_when_no_boundary": (
                    candidate_output == parent["output"] if stop_reason is None else None
                ),
                "source": {
                    "passed": bool(source_row["passed"]),
                    "score": float(source_row["score"]),
                    "output_sha256": str(source_row["output_sha256"]),
                },
                "source_passing_regression": bool(
                    source_row["passed"] and not candidate_passed
                ),
                "generation_error": None,
            }
        )
        if ordinal % 100 == 0:
            print(
                json.dumps(
                    {
                        "evaluated": ordinal,
                        "candidate_functional": sum(
                            row["candidate_functional_pass"] for row in rows
                        ),
                        "parent_functional": sum(
                            row["parent_functional_pass"] for row in rows
                        ),
                        "source_functional": sum(row["source"]["passed"] for row in rows),
                        "route_correct": sum(row["route_correct"] for row in rows),
                        "raw_collapses": sum(
                            row["candidate_raw_collapse"]["collapse_detected"]
                            for row in rows
                        ),
                    }
                ),
                flush=True,
            )

    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    candidate_functional = sum(row["candidate_functional_pass"] for row in rows)
    parent_functional = sum(row["parent_functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    by_capability = {
        capability: {
            "candidate_functional": sum(
                row["capability"] == capability and row["candidate_functional_pass"]
                for row in rows
            ),
            "parent_functional": sum(
                row["capability"] == capability and row["parent_functional_pass"]
                for row in rows
            ),
            "source_functional": sum(
                row["capability"] == capability and row["source"]["passed"]
                for row in rows
            ),
            "route_correct": sum(
                row["capability"] == capability and row["route_correct"] for row in rows
            ),
        }
        for capability in sorted(counts)
    }
    comparison = r49._bootstrap(
        [row["candidate_functional_pass"] for row in rows],
        [row["source"]["passed"] for row in rows],
    )
    metrics = {
        "rows": len(rows),
        "candidate_functional": candidate_functional,
        "parent_functional": parent_functional,
        "source_functional": source_functional,
        "source_passing_regressions": regressions,
        "source_retention": (source_functional - regressions) / source_functional,
        "candidate_minus_source": comparison,
        "candidate_raw_collapses": sum(
            row["candidate_raw_collapse"]["collapse_detected"] for row in rows
        ),
        "candidate_final_collapses": sum(
            row["candidate_final_collapse"]["collapse_detected"] for row in rows
        ),
        "generation_errors": sum(row["generation_error"] is not None for row in rows),
        "route_correct": sum(row["route_correct"] for row in rows),
        "physical_sparse_rows": sum(
            row["candidate_selected_route_only"]
            and row["candidate_maximum_cakes_called"] == 1
            for row in rows
        ),
        "no_boundary_parent_exact": sum(
            row["candidate_parent_exact_when_no_boundary"] is True for row in rows
        ),
        "no_boundary_parent_rows": sum(
            row["candidate_parent_exact_when_no_boundary"] is not None for row in rows
        ),
        "runtime_boundary_stops": sum(
            row["candidate_runtime_stop_reason"] is not None for row in rows
        ),
        "by_capability": by_capability,
        "wall_seconds": time.perf_counter() - started,
    }
    gates = {
        "matrix": len(rows) == EXPECTED_ROWS,
        "functional": candidate_functional >= 1_260,
        "parent_nondegradation": candidate_functional >= parent_functional,
        "source_noninferior_point": candidate_functional >= source_functional,
        "source_noninferior_bootstrap": comparison["lower_95"] >= -0.02,
        "source_retention": metrics["source_retention"] >= 0.94,
        "per_capability": all(
            value["candidate_functional"] >= 65 for value in by_capability.values()
        ),
        "zero_raw_collapse": metrics["candidate_raw_collapses"] == 0,
        "zero_final_collapse": metrics["candidate_final_collapses"] == 0,
        "zero_generation_error": metrics["generation_errors"] == 0,
        "route_exact": metrics["route_correct"] == EXPECTED_ROWS,
        "physical_sparse": metrics["physical_sparse_rows"] == EXPECTED_ROWS,
        "parent_parity_below_boundary": (
            metrics["no_boundary_parent_exact"] == metrics["no_boundary_parent_rows"]
        ),
        "artifacts_unchanged": all(sha256_file(path) == digest for path, digest in frozen),
    }
    if not math.isfinite(metrics["wall_seconds"]) or metrics["wall_seconds"] <= 0:
        raise ScreenError("R60 screen timing invalid")
    passed = all(gates.values())
    result = {
        "format": "abi-r60-prospective-frozen-endpoint-screen/1",
        "verdict": (
            "PASS_R60_PROSPECTIVE_FROZEN_ENDPOINT"
            if passed
            else "FAIL_R60_PROSPECTIVE_FROZEN_ENDPOINT"
        ),
        "catalog_sha256": CATALOG_SHA256,
        "candidate_checkpoint_sha256": r55.CANDIDATE_SHA256,
        "candidate_metadata_sha256": r55.METADATA_SHA256,
        "router_sha256": r55.ROUTER_SHA256,
        "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
        "source_outputs_sha256": SOURCE_OUTPUTS_SHA256,
        "metrics": metrics,
        "gates": gates,
        "runtime_telemetry": telemetry,
        "artifacts": {
            "evaluation": {
                "path": raw_path.name,
                "sha256": sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            }
        },
        "training_steps": 0,
        "teacher_calls": 0,
        "teacher_present_at_inference": False,
        "prospective_promotion_eligible": passed,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--catalog-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.candidate.resolve(),
        args.router.resolve(),
        args.catalog.resolve(),
        args.source_dir.resolve(),
        args.catalog_lock.resolve(),
        args.source_lock.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()

