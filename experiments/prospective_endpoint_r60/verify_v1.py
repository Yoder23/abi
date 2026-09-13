"""Fail-closed raw recomputation and live replay of the R60 prospective failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.irreversible_collapse_invariant_r59 import screen_v1 as r59
from experiments.isolated_capability_cakes_r53 import fit_router
from experiments.prospective_endpoint_r60 import screen_v1 as screen


RESULT_SHA256 = "755c285d52bf50b3a6826ef5112cc111346740e04b8f75f74e15fa60d6ef4140"
RAW_SHA256 = "e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf"


class VerificationError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VerificationError(f"required JSON absent: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"required JSON unreadable: {path}") from error
    if not isinstance(value, dict):
        raise VerificationError(f"required JSON is not an object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise VerificationError(f"required JSONL absent: {path}")
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"required JSONL unreadable: {path}") from error
    if any(not isinstance(row, dict) for row in values):
        raise VerificationError(f"required JSONL contains non-object: {path}")
    return values


def verify(
    run_dir: Path,
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
        raise VerificationError(f"immutable verification exists: {output}")
    result_path = run_dir / "result.json"
    raw_path = run_dir / "evaluation.jsonl"
    if sha256_file(result_path) != RESULT_SHA256 or sha256_file(raw_path) != RAW_SHA256:
        raise VerificationError("R60 frozen result or raw rows changed")
    result = _json(result_path)
    unsigned = dict(result)
    stored_evidence = unsigned.pop("evidence_sha256", None)
    if not isinstance(stored_evidence, str) or evidence_hash(unsigned) != stored_evidence:
        raise VerificationError("R60 result evidence digest changed")
    if (
        result.get("format") != "abi-r60-prospective-frozen-endpoint-screen/1"
        or result.get("verdict") != "FAIL_R60_PROSPECTIVE_FROZEN_ENDPOINT"
        or result.get("prospective_promotion_eligible") is not False
        or result.get("full_abi_moonshot") != "OPEN"
    ):
        raise VerificationError("R60 result scope changed")
    artifact = result.get("artifacts", {}).get("evaluation", {})
    if artifact != {
        "path": raw_path.name,
        "sha256": RAW_SHA256,
        "bytes": raw_path.stat().st_size,
    }:
        raise VerificationError("R60 result/raw binding changed")
    frozen = (
        (candidate / "model.safetensors", r55.CANDIDATE_SHA256),
        (candidate / "metadata.json", r55.METADATA_SHA256),
        (router_path, r55.ROUTER_SHA256),
        (catalog_path, screen.CATALOG_SHA256),
        (catalog_lock, screen.CATALOG_LOCK_SHA256),
        (source_lock, screen.SOURCE_LOCK_SHA256),
        (source_dir / "receipt.json", screen.SOURCE_RECEIPT_SHA256),
        (source_dir / "source_outputs.jsonl", screen.SOURCE_OUTPUTS_SHA256),
    )
    for path, digest in frozen:
        if not path.is_file() or sha256_file(path) != digest:
            raise VerificationError(f"R60 frozen input changed: {path}")
    r55._preflight(candidate)
    if not torch.cuda.is_available():
        raise VerificationError("R60 live verification requires CUDA")

    probes = list(load_probe_catalog(catalog_path)["probes"])
    rows = _jsonl(raw_path)
    source_rows = _jsonl(source_dir / "source_outputs.jsonl")
    source = {str(row["probe_id"]): row for row in source_rows}
    if (
        len(probes) != screen.EXPECTED_ROWS
        or len(rows) != screen.EXPECTED_ROWS
        or len(source) != screen.EXPECTED_ROWS
        or [row.get("probe_id") for row in rows]
        != [probe["probe_id"] for probe in probes]
    ):
        raise VerificationError("R60 matrix identity changed")

    router = torch.nn.Linear(r49.FEATURES, len(fit_router.CAPABILITY_TO_INDEX))
    router.load_state_dict(load_file(str(router_path), device="cpu"), strict=True)
    router.eval()
    device = torch.device("cuda")
    model, tokenizer, _ = load_layercake_core(
        candidate, layercake_root=layercake_root, device=device
    )
    model.eval()
    telemetry = {
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
    live_exact = 0
    for ordinal, (probe, row) in enumerate(zip(probes, rows, strict=True), 1):
        prompt = str(probe["prompt"])
        route = int(router(r49._feature(prompt)).argmax())
        expected_route = fit_router.CAPABILITY_TO_INDEX[str(probe["capability"])]
        source_row = source[str(probe["probe_id"])]
        parent = screen._generate_parent_raw(
            model, tokenizer, prompt, route, int(probe["max_new_tokens"]), device
        )
        candidate_output, candidate_tokens, _, candidate_physical = r59._generate(
            model,
            tokenizer,
            prompt,
            route,
            int(probe["max_new_tokens"]),
            device,
            telemetry=telemetry,
        )
        candidate_raw_ids, stop_reason = screen._candidate_prefix(
            parent["raw_output_token_ids"]
        )
        candidate_raw_output = tokenizer.decode(
            candidate_raw_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        derived = _truncate_novel_lexical_repetition(
            candidate_raw_output, prompt, threshold=1
        )
        candidate_passed, candidate_score = evaluate_output(
            candidate_output, probe["evaluator"]
        )
        parent_passed, parent_score = evaluate_output(
            parent["output"], probe["evaluator"]
        )
        raw_collapse = _collapse_metrics(
            candidate_raw_ids,
            candidate_raw_output,
            tokenizer.encode(prompt + "\n"),
            prompt,
        )
        final_collapse = _collapse_metrics(
            candidate_tokens,
            candidate_output,
            tokenizer.encode(prompt + "\n"),
            prompt,
        )
        if (
            row.get("capability") != probe["capability"]
            or row.get("prompt") != prompt
            or row.get("prompt_sha256")
            != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            or row.get("evaluator") != probe["evaluator"]
            or row.get("route") != route
            or row.get("expected_route") != expected_route
            or row.get("route_correct") is not (route == expected_route)
            or candidate_output != derived
            or row.get("candidate_output") != candidate_output
            or row.get("candidate_output_token_ids") != candidate_tokens
            or row.get("candidate_raw_output") != candidate_raw_output
            or row.get("candidate_raw_output_token_ids") != candidate_raw_ids
            or row.get("candidate_runtime_stop_reason") != stop_reason
            or row.get("candidate_functional_pass") is not bool(candidate_passed)
            or row.get("candidate_functional_score") != float(candidate_score)
            or row.get("candidate_raw_collapse") != raw_collapse
            or row.get("candidate_final_collapse") != final_collapse
            or row.get("candidate_maximum_cakes_called") != max(map(len, candidate_physical))
            or row.get("candidate_selected_route_only")
            is not all(calls == (route,) for calls in candidate_physical)
            or row.get("parent_output") != parent["output"]
            or row.get("parent_output_token_ids") != parent["output_token_ids"]
            or row.get("parent_raw_output") != parent["raw_output"]
            or row.get("parent_raw_output_token_ids") != parent["raw_output_token_ids"]
            or row.get("parent_functional_pass") is not bool(parent_passed)
            or row.get("parent_functional_score") != float(parent_score)
            or row.get("parent_lexical_postprocessor_changed_output")
            is not parent["lexical_postprocessor_changed_output"]
            or row.get("parent_language_model_eos") is not parent["language_model_eos"]
            or row.get("source")
            != {
                "passed": bool(source_row["passed"]),
                "score": float(source_row["score"]),
                "output_sha256": str(source_row["output_sha256"]),
            }
            or row.get("source_passing_regression")
            is not bool(source_row["passed"] and not candidate_passed)
            or row.get("generation_error") is not None
            or not isinstance(row.get("candidate_latency_seconds"), (int, float))
            or not math.isfinite(row["candidate_latency_seconds"])
            or row["candidate_latency_seconds"] <= 0
            or not isinstance(row.get("parent_latency_seconds"), (int, float))
            or not math.isfinite(row["parent_latency_seconds"])
            or row["parent_latency_seconds"] <= 0
        ):
            raise VerificationError(f"R60 live row changed: {probe['probe_id']}")
        no_boundary_exact = candidate_output == parent["output"] if stop_reason is None else None
        if row.get("candidate_parent_exact_when_no_boundary") is not no_boundary_exact:
            raise VerificationError(f"R60 parity row changed: {probe['probe_id']}")
        live_exact += 1
        if ordinal % 100 == 0:
            print(json.dumps({"live_replayed": ordinal, "exact": live_exact}), flush=True)

    candidate_functional = sum(row["candidate_functional_pass"] for row in rows)
    parent_functional = sum(row["parent_functional_pass"] for row in rows)
    source_functional = sum(row["source"]["passed"] for row in rows)
    regressions = sum(row["source_passing_regression"] for row in rows)
    comparison = r49._bootstrap(
        [row["candidate_functional_pass"] for row in rows],
        [row["source"]["passed"] for row in rows],
    )
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
        for capability in sorted(fit_router.CAPABILITY_TO_INDEX)
    }
    stored_metrics = result["metrics"]
    recomputed = {
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
        "physical_sparse_rows": screen.EXPECTED_ROWS,
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
        "wall_seconds": stored_metrics["wall_seconds"],
    }
    if recomputed != stored_metrics or telemetry != result.get("runtime_telemetry"):
        raise VerificationError("R60 aggregate recomputation changed")
    gates = {
        "matrix": len(rows) == screen.EXPECTED_ROWS,
        "functional": candidate_functional >= 1_260,
        "parent_nondegradation": candidate_functional >= parent_functional,
        "source_noninferior_point": candidate_functional >= source_functional,
        "source_noninferior_bootstrap": comparison["lower_95"] >= -0.02,
        "source_retention": recomputed["source_retention"] >= 0.94,
        "per_capability": all(
            value["candidate_functional"] >= 65 for value in by_capability.values()
        ),
        "zero_raw_collapse": recomputed["candidate_raw_collapses"] == 0,
        "zero_final_collapse": recomputed["candidate_final_collapses"] == 0,
        "zero_generation_error": recomputed["generation_errors"] == 0,
        "route_exact": recomputed["route_correct"] == screen.EXPECTED_ROWS,
        "physical_sparse": recomputed["physical_sparse_rows"] == screen.EXPECTED_ROWS,
        "parent_parity_below_boundary": (
            recomputed["no_boundary_parent_exact"]
            == recomputed["no_boundary_parent_rows"]
        ),
        "artifacts_unchanged": all(sha256_file(path) == digest for path, digest in frozen),
    }
    if gates != result.get("gates") or all(gates.values()):
        raise VerificationError("R60 gate recomputation changed")
    receipt = {
        "format": "abi-r60-live-strict-failure-verification/1",
        "verdict": "PASS_VERIFICATION_OF_R60_PROSPECTIVE_FAILURE",
        "result_sha256": RESULT_SHA256,
        "raw_sha256": RAW_SHA256,
        "raw_rows_recomputed": len(rows),
        "live_parent_rows_exact": live_exact,
        "live_candidate_rows_exact": live_exact,
        "failed_gates": sorted(key for key, value in gates.items() if not value),
        "candidate_functional": candidate_functional,
        "source_functional": source_functional,
        "source_retention": recomputed["source_retention"],
        "full_abi_moonshot": "OPEN",
    }
    write_json_once(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--catalog-lock", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(
        args.run_dir.resolve(),
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

