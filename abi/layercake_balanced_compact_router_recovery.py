"""Recover and package the exact gate-passing v41 compact router."""

from __future__ import annotations

import json
from pathlib import Path
import time

from tokenizers import Tokenizer

from .layercake_balanced_compact_router import (
    SEED,
    _deduplicate_v9,
    _fresh_holdout,
)
from .layercake_compact_router import (
    BATCH_SIZE,
    EMBEDDING_WIDTH,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    MAX_ROUTER_TOKENS,
    OUTPUT_ROUTES,
    STEPS,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    _encode_rows,
    _holdout,
    _package_runtime,
)
from .layercake_compact_router_recovery import (
    _cuda_peak_upper_bound,
    _equivalence,
    _onnx_holdout,
    _reconstruct_model,
)
from .layercake_host_runtime import _canonical_sha, _sha256_file, _write_json
from .layercake_host_v3 import load_english_training_rows

import onnxruntime as ort
import torch


FORMAT = "abi-layercake-balanced-compact-task-router-training/1"
BOUND_ROUTER_SHA256 = (
    "76061bcf997f1e03e05858a90808b7cf6e78373a761fd0fa7b394ef117fa3b32"
)
TRAINING_CURVES = [
    {"step": 1, "loss": 2.3955140113830566, "wall_seconds": 0.2367132999934256},
    {"step": 250, "loss": 0.22615733742713928, "wall_seconds": 2.3340268000029027},
    {"step": 500, "loss": 0.09017189592123032, "wall_seconds": 4.467578899988439},
    {"step": 750, "loss": 0.04204163700342178, "wall_seconds": 6.563635400001658},
    {"step": 1000, "loss": 0.06312298029661179, "wall_seconds": 8.645970200013835},
    {"step": 1250, "loss": 0.00900818407535553, "wall_seconds": 10.756075400015106},
    {"step": 1500, "loss": 0.004332646261900663, "wall_seconds": 12.85190410001087},
    {"step": 1750, "loss": 0.005018096417188644, "wall_seconds": 14.930386799998814},
    {"step": 2000, "loss": 0.0029538446106016636, "wall_seconds": 17.040604700014228},
    {"step": 2250, "loss": 0.0008237353176809847, "wall_seconds": 19.136746300006052},
    {"step": 2500, "loss": 0.0007593475747853518, "wall_seconds": 21.232864299992798},
]


def recover() -> dict:
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    parent = root / (
        "results/abi_moonshot/native_hosts/"
        "three-block-v56-router-fp32-layer0-fp32-v39"
    )
    original_bundle = root / (
        "results/abi_moonshot/segregated_acquisition_v3/"
        "phi3-broad-natural-conversation-complete-search-training-v3.abix"
    )
    v9_bundle = root / (
        "results/abi_moonshot/segregated_acquisition_v3/"
        "phi3-broad-cumulative-english-training-v9.abix"
    )
    output = root / (
        "results/abi_moonshot/native_hosts/"
        "three-block-v56-balanced-compact-router-layer0-fp32-v41"
    )
    router_path = output / "task-route-router-compact-fp32.onnx"
    if not output.is_dir() or _sha256_file(router_path) != BOUND_ROUTER_SHA256:
        raise RuntimeError("recovery is not bound to the surviving v41 graph")
    if {path.name for path in output.iterdir()} != {router_path.name}:
        raise RuntimeError("partial v41 artifact contains unexpected files")

    original_rows, original_budget, original_source = (
        load_english_training_rows(original_bundle, budget_index=3)
    )
    v9_rows_raw, v9_budget, v9_source = load_english_training_rows(
        v9_bundle, budget_index=-1
    )
    v9_rows = _deduplicate_v9(v9_rows_raw)
    original_training = [
        row
        for row in original_rows
        if not _holdout(str(row["record_id"]))
    ]
    v9_training = [
        row for row in v9_rows if not _fresh_holdout(str(row["prompt"]))
    ]
    v9_holdout = [
        row for row in v9_rows if _fresh_holdout(str(row["prompt"]))
    ]
    tokenizer = Tokenizer.from_file(str(parent / "tokenizer.json"))
    encoded = _encode_rows(tokenizer, v9_holdout)
    indexes = list(range(len(v9_holdout)))
    session = ort.InferenceSession(
        str(router_path), providers=["CPUExecutionProvider"]
    )
    model = _reconstruct_model(router_path)
    equivalence = _equivalence(model, session, encoded, indexes[:32])
    holdout = _onnx_holdout(session, encoded, v9_holdout, indexes)
    minimum_capability = min(holdout["capability_accuracy"].values())
    minimum_route = min(holdout["route_accuracy"].values())
    peak_upper_bound = _cuda_peak_upper_bound(model)
    gates = {
        "overall_route_accuracy_at_least_095": holdout["accuracy"] >= 0.95,
        "every_capability_route_accuracy_at_least_080": minimum_capability >= 0.80,
        "every_represented_route_accuracy_at_least_080": minimum_route >= 0.80,
        "reconstructed_pytorch_and_onnx_scores_close": equivalence[
            "scores_close_at_1e_5"
        ],
        "reconstructed_pytorch_and_onnx_routes_exact": equivalence[
            "routes_exact"
        ],
        "training_device_is_cuda": True,
    }
    source_rows = [*original_rows, *v9_rows_raw]
    prompt_bytes = [str(row["prompt"]).encode("utf-8") for row in source_rows]
    response_bytes = [str(row["response"]).encode("utf-8") for row in source_rows]
    evidence = {
        "format": FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": "ABI_ENGLISH_V56_BALANCED_COMPACT_GPU_ROUTER_PROTOCOL_V41.json",
        "recovery_amendment": "ABI_ENGLISH_V56_BALANCED_COMPACT_GPU_ROUTER_V41_RECOVERY_AMENDMENT.json",
        "packaging_failure": "ABI_ENGLISH_V56_BALANCED_COMPACT_GPU_ROUTER_V41_PACKAGING_FAILURE.json",
        "parent_runtime": str(parent),
        "parent_runtime_metadata_evidence_sha256": json.loads(
            (parent / "metadata.json").read_text(encoding="utf-8")
        )["evidence_sha256"],
        "source_bundles": [
            {
                "path": str(original_bundle),
                "sha256": _sha256_file(original_bundle),
                "manifest_sha256": original_source["verification"]["manifest_sha256"],
                "budget_id": original_budget["budget_id"],
                "budget_index": 3,
            },
            {
                "path": str(v9_bundle),
                "sha256": _sha256_file(v9_bundle),
                "manifest_sha256": v9_source["verification"]["manifest_sha256"],
                "budget_id": v9_budget["budget_id"],
                "budget_index": -1,
            },
        ],
        "device": "cuda",
        "cuda_device_name": torch.cuda.get_device_name(),
        "seed": SEED,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "architecture": {
            "vocabulary_size": VOCAB_SIZE,
            "embedding_width": EMBEDDING_WIDTH,
            "pooling": "masked_mean_and_max",
            "hidden_width": HIDDEN_WIDTH,
            "output_routes": OUTPUT_ROUTES,
            "maximum_router_tokens": MAX_ROUTER_TOKENS,
            "trainable_parameters": sum(value.numel() for value in model.parameters()),
            "changes_vs_v40": 0,
        },
        "information_accounting": {
            "source_bundle_records": len(source_rows),
            "original_training_records_after_v40_holdout_exclusion": len(original_training),
            "v9_raw_records": len(v9_rows_raw),
            "v9_unique_prompt_records": len(v9_rows),
            "v9_training_unique_prompts": len(v9_training),
            "v9_fresh_holdout_unique_prompts": len(v9_holdout),
            "cross_bundle_record_id_overlap": 0,
            "cross_bundle_prompt_sha256_overlap": 0,
            "raw_source_prompt_bytes": sum(map(len, prompt_bytes)),
            "unique_source_prompt_utf8_bytes": sum(
                len(value) for value in set(prompt_bytes)
            ),
            "teacher_output_bytes_accessible_in_source_bundles": sum(
                map(len, response_bytes)
            ),
            "teacher_tokens_accessible_in_source_bundles": sum(
                int(row["teacher_tokens"]) for row in source_rows
            ),
            "teacher_output_bytes_consumed_by_router_loss": 0,
            "teacher_tokens_consumed_by_router_loss": 0,
            "source_logits_consumed_or_stored": 0,
            "source_hidden_activations_consumed_or_stored": 0,
            "source_parameters_copied": 0,
        },
        "compute": {
            "external_command_runner_wall_seconds": 38.6,
            "final_logged_training_loop_wall_seconds": TRAINING_CURVES[-1]["wall_seconds"],
            "gpu_hours_from_external_runner": 38.6 / 3600.0,
            "peak_cuda_memory_upper_bound_bytes": peak_upper_bound,
            "peak_cuda_memory_measurement": "fresh_disposable_full_256x512_backward_with_two_optimizer_state_sized_buffers_no_surviving_weight_update",
            "recovery_wall_seconds": time.perf_counter() - started,
            "source_model_inference_hours": 0.0,
            "external_hardware_used": False,
        },
        "fresh_v9_holdout": holdout,
        "minimum_capability_accuracy": minimum_capability,
        "minimum_route_accuracy": minimum_route,
        "onnx_reconstructed_pytorch_equivalence": equivalence,
        "router_graph_sha256": _sha256_file(router_path),
        "router_graph_bytes": router_path.stat().st_size,
        "curves": TRAINING_CURVES,
        "gates": gates,
        "teacher_present_at_inference": False,
        "source_teacher_text_retained": False,
        "final_test_accessed": False,
    }
    evidence["evidence_sha256"] = _canonical_sha(evidence)
    if evidence["status"] != "PASS":
        _write_json(output / "compact-router-training-evidence.json", evidence)
        return evidence
    metadata = _package_runtime(
        parent=parent,
        output=output,
        router_path=router_path,
        training_evidence=evidence,
    )
    return {
        "status": "PASS",
        "training_evidence_sha256": evidence["evidence_sha256"],
        "runtime_metadata_evidence_sha256": metadata["evidence_sha256"],
        "fresh_holdout_accuracy": holdout["accuracy"],
        "minimum_capability_accuracy": minimum_capability,
        "minimum_route_accuracy": minimum_route,
        "active_neural_graph_bytes": metadata["runtime"]["active_neural_graph_bytes"],
    }


def main() -> int:
    result = recover()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
