"""Prospective live R84 deployment-mixture pointer screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from abi.layercake_core_loader import CAPABILITY_CAKE_ORDER, load_layercake_core
from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import (
    _prompt_identity_next_probabilities,
    _select_next_token,
    _sha256_file,
)
from experiments.foreign_capability_r14.core import (
    evidence_hash,
    write_json_once,
    write_jsonl_once,
)
from experiments.generic_prompt_identity_r83.screen_host_v1 import (
    _bootstrap,
    _jsonl,
)
from experiments.mixture_pointer_r84.extension_v1 import (
    BASE_ARCHITECTURE,
    GENERIC_ARCHITECTURE,
    GENERIC_SCOPE,
    load_r84_core,
)


CATALOG_SHA256 = "0d2b24417bf1ea925b97747d54ac7053ac7a38f517966431e8befb8f893d2c54"
SOURCE_RESULT_SHA256 = "474e2e027b7e2447f4a33174061ea1cf556dd0572b74f5730a8bcafe0f35078f"
SOURCE_RAW_SHA256 = "9a2783054bdf9f8d309ae18b268d53d5fe9dfefeb6c67301e7bdbf095e9e7a20"
SOURCE_EVIDENCE_SHA256 = "2a6dde5605be5683934896b134767c911f12e5573c96fa601e969a459b50b452"
PARENT_SHA256 = "b6977f087ac42e6e4234d026b4cd83827b720d8973cca049daf18bcc8b96a64e"
PARENT_METADATA_SHA256 = "1c91e94abc3f2faa9a6f7d68689451dc94098dd2330652a4713a0116c3080e0e"
MAIN_ARCHIVE_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
ANCHOR_ARCHIVE_SHA256 = "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
ROWS = 1_400
ROWS_PER_FAMILY = 200
ROUTE = CAPABILITY_CAKE_ORDER.index("domain_independent_reasoning")


class ScreenError(RuntimeError):
    pass


def _load_binding(
    path: Path, *, screen_path: Path, protocol_path: Path
) -> dict[str, Any]:
    if not path.is_file():
        raise ScreenError("candidate binding is absent")
    binding = json.loads(path.read_text(encoding="utf-8"))
    expected = binding.get("binding_sha256")
    payload = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if (
        binding.get("format") != "abi-r84-candidate-screen-binding/1"
        or expected != _canonical_sha(payload)
        or binding.get("candidate_generation_observations_before_binding") != 0
        or binding.get("screen_sha256") != _sha256_file(screen_path)
        or binding.get("protocol_sha256") != _sha256_file(protocol_path)
    ):
        raise ScreenError("candidate binding or frozen evaluator changed")
    return binding


def _preflight(
    candidate: Path, parent: Path, binding: dict[str, Any]
) -> dict[str, Any]:
    for path, digest in (
        (candidate / "model.safetensors", binding["candidate_checkpoint_sha256"]),
        (candidate / "metadata.json", binding["candidate_metadata_sha256"]),
        (parent / "model.safetensors", PARENT_SHA256),
        (parent / "metadata.json", PARENT_METADATA_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != digest:
            raise ScreenError(f"frozen host input changed: {path}")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    unsigned = dict(metadata)
    manifest_sha = unsigned.pop("manifest_sha256", None)
    training = metadata.get("training", {})
    imported = metadata.get("imported_artifact", {})
    anchor = metadata.get("broad_behavior_anchor", {})
    boundary = metadata.get("foreign_source_boundary", {})
    acquired = metadata.get("acquired_core", {})
    pointer = acquired.get("prompt_identity_carriage") or {}
    expansion = acquired.get("capability_cake_expansion") or {}
    architecture = metadata.get("architecture", {})
    recovery = training.get("self_generated_prefix_recovery", {})
    extension = metadata.get("r84_additive_extension", {})
    if (
        _manifest_sha(unsigned) != manifest_sha
        or metadata.get("status")
        != "TRAINED_NOT_YET_SEMANTICALLY_OR_OPERATIONALLY_CERTIFIED"
        or training.get("seed") != 84_001
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("batch_size") != 4
        or training.get("anchor_batch_size") != 4
        or training.get("trainable_scope") != GENERIC_SCOPE
        or training.get("classifier_loss_weight") != 0.0
        or training.get("prompt_overlap_loss_weight") != 1.0
        or training.get("prompt_identity_loss_weight") != 1.0
        or training.get("source_teacher_forward_tokens") != 0
        or training.get("source_model_inference_seconds") != 0.0
        or recovery.get("start_step") != 400
        or recovery.get("interval") != 8
        or recovery.get("horizons") != [4, 8, 16]
        or imported.get("archive_sha256_after") != MAIN_ARCHIVE_SHA256
        or imported.get("selected_english_records") != 8_129
        or imported.get("all_selected_records_seen") is not True
        or anchor.get("archive_sha256_after") != ANCHOR_ARCHIVE_SHA256
        or anchor.get("selected_english_records") != 24_419
        or boundary.get("teacher_present_at_inference") is not False
        or boundary.get("source_parameters_copied") != 0
        or boundary.get("source_transformer_blocks_retained") != 0
        or architecture.get("architecture_version") != BASE_ARCHITECTURE
        or architecture.get("layers") != 6
        or architecture.get("task_cakes") != 14
        or architecture.get("capability_adapter_rank") != 32
        or extension.get("architecture") != GENERIC_ARCHITECTURE
        or extension.get("deployed_mixture_nll_trained") is not True
        or pointer.get("parameter_count") != 49_935
        or pointer.get("selective_parent_top1_deficit_labels") is not False
        or pointer.get("parent_state_preserved_exact") is not True
        or expansion.get("installed_deep_adapters") != 84
        or expansion.get("maximum_active_deep_adapters_per_sequence") != 6
        or expansion.get("maximum_active_capability_cakes_per_sequence") != 1
        or acquired.get("physical_sparse_topology_preserved") is not True
    ):
        raise ScreenError("R84 acquisition or deployment contract changed")
    return metadata


def _check_sparse(model: Any) -> None:
    if tuple(model.last_cake_calls) != (ROUTE,):
        raise ScreenError("host executed an inactive capability cake")
    traces = [
        tuple(getattr(block, "_abi_last_deep_adapter_routes", ()))
        for block in model.transformer.h
    ]
    if len(traces) != 6 or any(trace != (ROUTE,) for trace in traces):
        raise ScreenError("host executed an inactive or missing deep adapter")


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    maximum: int,
    device: torch.device,
    *,
    require_pointer: bool,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt + "\n")
    if len(prompt_ids) + maximum > int(model.config.max_tokens):
        raise ScreenError("R84 prompt exceeds host context")
    route = torch.tensor([ROUTE], dtype=torch.long, device=device)
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
        task_routes=route,
        use_cache=True,
    )
    _check_sparse(model)
    pointer = getattr(model, "_abi_prompt_identity_bridge", None)
    if (pointer is not None) != require_pointer:
        raise ScreenError("prompt pointer presence changed")
    prompt_tokens = ids[0]
    prompt_keys = (
        pointer.key(result["hidden"][0, : len(prompt_ids)])
        if pointer is not None
        else None
    )
    next_hidden = result["hidden"][:, -1]
    state = {
        "past_key_values": result["past_key_values"],
        "next_logits": result["logits"][:, -1],
    }
    generated: list[int] = []
    invocations = 1
    pointer_invocations = 0
    eos = False
    for _ in range(maximum):
        if pointer is not None:
            scores = _prompt_identity_next_probabilities(
                logits=state["next_logits"],
                query_hidden=next_hidden,
                prompt_keys=prompt_keys,
                prompt_ids=prompt_tokens,
                route=route,
                bridge=pointer,
            )
            pointer_invocations += 1
        else:
            scores = state["next_logits"][0]
        selected = _select_next_token(
            scores,
            generated=generated,
            no_repeat_ngram_size=0,
        ).to(device)
        token = int(selected.item())
        if token == tokenizer.eos_token_id:
            eos = True
            break
        generated.append(token)
        result = model(
            selected[:, None],
            task_routes=route,
            past_key_values=state["past_key_values"],
            use_cache=True,
        )
        invocations += 1
        _check_sparse(model)
        state = {
            "past_key_values": result["past_key_values"],
            "next_logits": result["logits"][:, -1],
        }
        next_hidden = result["hidden"][:, -1]
    output = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "output": output,
        "token_ids": generated,
        "latency_seconds": time.perf_counter() - started,
        "language_model_eos": eos,
        "model_invocations": invocations,
        "task_cake_invocations": invocations,
        "deep_adapter_invocations": invocations * 6,
        "pointer_invocations": pointer_invocations,
        "physical_sparse": True,
    }


@torch.inference_mode()
def _candidate_score(
    model: Any,
    tokenizer: Any,
    prompt: str,
    candidate: str,
    device: torch.device,
) -> float:
    prompt_ids = tokenizer.encode(prompt + "\n")
    candidate_ids = tokenizer.encode(candidate)
    ids = torch.tensor([prompt_ids + candidate_ids], dtype=torch.long, device=device)
    route = torch.tensor([ROUTE], dtype=torch.long, device=device)
    result = model(
        ids,
        prompt_lengths=torch.tensor([len(prompt_ids)], device=device),
        task_routes=route,
        use_cache=False,
    )
    _check_sparse(model)
    pointer = getattr(model, "_abi_prompt_identity_bridge", None)
    if pointer is None:
        raise ScreenError("candidate diagnostic lost its pointer")
    prompt_tokens = ids[0, : len(prompt_ids)]
    prompt_keys = pointer.key(result["hidden"][0, : len(prompt_ids)])
    values = []
    for offset, target in enumerate(candidate_ids):
        preceding = len(prompt_ids) + offset - 1
        probability = _prompt_identity_next_probabilities(
            logits=result["logits"][:, preceding],
            query_hidden=result["hidden"][:, preceding],
            prompt_keys=prompt_keys,
            prompt_ids=prompt_tokens,
            route=route,
            bridge=pointer,
        )[target]
        values.append(math.log(max(float(probability.item()), 1e-30)))
    return sum(values) / len(values)


def run(
    *,
    candidate: Path,
    parent: Path,
    binding_path: Path,
    screen_path: Path,
    protocol_path: Path,
    layercake_root: Path,
    catalog_path: Path,
    source_result_path: Path,
    source_raw_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ScreenError(f"immutable R84 screen exists: {output}")
    for path, digest in (
        (catalog_path, CATALOG_SHA256),
        (source_result_path, SOURCE_RESULT_SHA256),
        (source_raw_path, SOURCE_RAW_SHA256),
    ):
        if not path.is_file() or _sha256_file(path) != digest:
            raise ScreenError("R81 catalog or source evidence changed")
    binding = _load_binding(
        binding_path, screen_path=screen_path, protocol_path=protocol_path
    )
    metadata = _preflight(candidate, parent, binding)
    source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    unsigned_source = dict(source_result)
    unsigned_source.pop("evidence_sha256", None)
    if (
        source_result.get("verdict") != "PASS_R81_SOURCE"
        or source_result.get("evidence_sha256") != SOURCE_EVIDENCE_SHA256
        or evidence_hash(unsigned_source) != SOURCE_EVIDENCE_SHA256
        or source_result.get("metrics", {}).get("prior_corrected_passing") != 1_382
    ):
        raise ScreenError("R81 source result is invalid or stale")
    probes = list(load_probe_catalog(catalog_path)["probes"])
    source_rows = {str(row["probe_id"]): row for row in _jsonl(source_raw_path)}
    if (
        len(probes) != ROWS
        or len(source_rows) != ROWS
        or set(source_rows) != {str(row["probe_id"]) for row in probes}
    ):
        raise ScreenError("R81 source coverage changed")
    for probe in probes:
        source = source_rows[str(probe["probe_id"])]
        if (
            source.get("prompt_sha256")
            != hashlib.sha256(str(probe["prompt"]).encode()).hexdigest()
            or source.get("expected_output_sha256")
            != hashlib.sha256(str(probe["evaluator"]["value"]).encode()).hexdigest()
        ):
            raise ScreenError("R81 raw row does not bind its catalog target")
    if not torch.cuda.is_available():
        raise ScreenError("R84 live screen requires CUDA")
    device = torch.device("cuda")
    process = psutil.Process()
    observed: dict[str, dict[str, Any]] = {
        str(probe["probe_id"]): {"probe": probe} for probe in probes
    }

    parent_model, parent_tokenizer, _ = load_layercake_core(
        parent, layercake_root=layercake_root, device=device
    )
    parent_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _generate(
            parent_model,
            parent_tokenizer,
            str(probe["prompt"]),
            int(probe["max_new_tokens"]),
            device,
            require_pointer=False,
        )
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        observed[str(probe["probe_id"])]["parent"] = {
            **generated,
            "passed": bool(passed),
            "score": float(score),
        }
        if index % 200 == 0:
            print(json.dumps({"system": "parent", "evaluated": index}), flush=True)
    parent_seconds = time.perf_counter() - parent_started
    del parent_model
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    candidate_model, candidate_tokenizer, _ = load_r84_core(
        candidate, layercake_root=layercake_root, device=device
    )
    candidate_started = time.perf_counter()
    for index, probe in enumerate(probes, 1):
        generated = _generate(
            candidate_model,
            candidate_tokenizer,
            str(probe["prompt"]),
            int(probe["max_new_tokens"]),
            device,
            require_pointer=True,
        )
        passed, score = evaluate_output(generated["output"], probe["evaluator"])
        collapse = _collapse_metrics(
            generated["token_ids"],
            generated["output"],
            candidate_tokenizer.encode(str(probe["prompt"]) + "\n"),
            str(probe["prompt"]),
        )
        scores = [
            _candidate_score(
                candidate_model,
                candidate_tokenizer,
                str(probe["prompt"]),
                str(value),
                device,
            )
            for value in probe["conditional_candidates"]
        ]
        selected = max(range(3), key=scores.__getitem__)
        observed[str(probe["probe_id"])]["candidate"] = {
            **generated,
            "passed": bool(passed),
            "score": float(score),
            "collapse": collapse,
            "conditional_scores": scores,
            "conditional_selected_index": selected,
            "conditional_passed": selected == int(probe["destination_display_index"]),
        }
        if index % 200 == 0:
            print(json.dumps({"system": "candidate", "evaluated": index}), flush=True)
    candidate_seconds = time.perf_counter() - candidate_started

    rows = []
    for probe in probes:
        probe_id = str(probe["probe_id"])
        source = source_rows[probe_id]
        parent_row = observed[probe_id]["parent"]
        candidate_row = observed[probe_id]["candidate"]
        rows.append(
            {
                "probe_id": probe_id,
                "premise_family": int(source["premise_family"]),
                "prompt_sha256": source["prompt_sha256"],
                "source_passed": bool(source["prior_corrected_passed"]),
                "parent_output": parent_row["output"],
                "parent_passed": parent_row["passed"],
                "candidate_output": candidate_row["output"],
                "candidate_token_ids": candidate_row["token_ids"],
                "candidate_passed": candidate_row["passed"],
                "candidate_collapse": candidate_row["collapse"],
                "candidate_model_invocations": candidate_row["model_invocations"],
                "candidate_task_cake_invocations": candidate_row[
                    "task_cake_invocations"
                ],
                "candidate_deep_adapter_invocations": candidate_row[
                    "deep_adapter_invocations"
                ],
                "candidate_pointer_invocations": candidate_row["pointer_invocations"],
                "candidate_physical_sparse": candidate_row["physical_sparse"],
                "conditional_scores": candidate_row["conditional_scores"],
                "conditional_selected_index": candidate_row[
                    "conditional_selected_index"
                ],
                "conditional_passed": candidate_row["conditional_passed"],
                "source_passing_retained": bool(
                    source["prior_corrected_passed"] and candidate_row["passed"]
                ),
            }
        )
    output.mkdir(parents=True)
    raw_path = output / "evaluation.jsonl"
    write_jsonl_once(raw_path, rows)
    source_passing = sum(row["source_passed"] for row in rows)
    candidate_passing = sum(row["candidate_passed"] for row in rows)
    parent_passing = sum(row["parent_passed"] for row in rows)
    retained = sum(row["source_passing_retained"] for row in rows)
    family = {
        str(index): {
            "rows": ROWS_PER_FAMILY,
            "source_passing": sum(
                row["premise_family"] == index and row["source_passed"] for row in rows
            ),
            "candidate_passing": sum(
                row["premise_family"] == index and row["candidate_passed"]
                for row in rows
            ),
            "parent_passing": sum(
                row["premise_family"] == index and row["parent_passed"] for row in rows
            ),
        }
        for index in range(7)
    }
    metrics = {
        "rows": ROWS,
        "source_passing": source_passing,
        "candidate_passing": candidate_passing,
        "parent_passing": parent_passing,
        "source_passing_retained": retained,
        "source_retention": retained / source_passing,
        "candidate_collapses": sum(
            row["candidate_collapse"]["collapse_detected"] for row in rows
        ),
        "conditional_passing": sum(row["conditional_passed"] for row in rows),
        "candidate_physical_sparse_rows": sum(
            row["candidate_physical_sparse"] for row in rows
        ),
        "by_family": family,
        "candidate_minus_source": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["source_passed"] for row in rows],
            84_001,
        ),
        "candidate_minus_parent": _bootstrap(
            [row["candidate_passed"] for row in rows],
            [row["parent_passed"] for row in rows],
            84_002,
        ),
        "parent_generation_seconds": parent_seconds,
        "candidate_generation_and_diagnostic_seconds": candidate_seconds,
        "candidate_peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "process_rss_bytes": int(process.memory_info().rss),
    }
    gates = {
        "matrix": len(rows) == ROWS,
        "source_quality": source_passing >= 1_330,
        "candidate_quality": candidate_passing >= 1_330,
        "candidate_family_floor": min(
            value["candidate_passing"] for value in family.values()
        )
        >= 180,
        "source_retention": metrics["source_retention"] >= 0.95,
        "causal_parent_gain": (candidate_passing - parent_passing) / ROWS >= 0.50,
        "zero_collapse": metrics["candidate_collapses"] == 0,
        "physical_sparse": metrics["candidate_physical_sparse_rows"] == ROWS,
        "source_absent_at_inference": metadata["foreign_source_boundary"][
            "teacher_present_at_inference"
        ]
        is False,
        "artifacts_unchanged": all(
            _sha256_file(path) == digest
            for path, digest in (
                (candidate / "model.safetensors", binding["candidate_checkpoint_sha256"]),
                (candidate / "metadata.json", binding["candidate_metadata_sha256"]),
                (parent / "model.safetensors", PARENT_SHA256),
                (parent / "metadata.json", PARENT_METADATA_SHA256),
            )
        ),
    }
    if not all(math.isfinite(float(value)) for value in (parent_seconds, candidate_seconds)):
        raise ScreenError("R84 runtime accounting is non-finite")
    passed = all(gates.values())
    result = {
        "format": "abi-r84-mixture-pointer-host-screen/1",
        "verdict": (
            "PASS_R84_BOUNDED_MIXTURE_POINTER_TRANSFER"
            if passed
            else "FAIL_R84_MIXTURE_POINTER_TRANSFER"
        ),
        "candidate_checkpoint_sha256": binding["candidate_checkpoint_sha256"],
        "candidate_metadata_sha256": binding["candidate_metadata_sha256"],
        "candidate_binding_sha256": binding["binding_sha256"],
        "parent_checkpoint_sha256": PARENT_SHA256,
        "catalog_sha256": CATALOG_SHA256,
        "source_result_sha256": SOURCE_RESULT_SHA256,
        "source_raw_sha256": SOURCE_RAW_SHA256,
        "metrics": metrics,
        "gates": gates,
        "artifacts": {
            "evaluation": {
                "path": raw_path.name,
                "sha256": _sha256_file(raw_path),
                "bytes": raw_path.stat().st_size,
            }
        },
        "teacher_present_at_inference": False,
        "source_parameters_retained": 0,
        "candidate_accessed_only_after_protocol_and_screen_code_frozen": True,
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": (
            "Bounded teacher-to-artifact-to-frozen-LayerCake acquisition for one "
            "prospective nonce relation/copy task; not unrestricted English or domains."
        ),
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "parent",
        "binding",
        "screen",
        "protocol",
        "layercake-root",
        "catalog",
        "source-result",
        "source-raw",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    run(
        candidate=args.candidate.resolve(),
        parent=args.parent.resolve(),
        binding_path=args.binding.resolve(),
        screen_path=args.screen.resolve(),
        protocol_path=args.protocol.resolve(),
        layercake_root=args.layercake_root.resolve(),
        catalog_path=args.catalog.resolve(),
        source_result_path=args.source_result.resolve(),
        source_raw_path=args.source_raw.resolve(),
        output=args.output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
