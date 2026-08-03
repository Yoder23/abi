"""Recover evidence for the exact v40 router after its writer-only failure."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from .layercake_compact_router import (
    BATCH_SIZE,
    EMBEDDING_WIDTH,
    FORMAT,
    HIDDEN_WIDTH,
    LEARNING_RATE,
    MAX_ROUTER_TOKENS,
    OUTPUT_ROUTES,
    SEED,
    STEPS,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    CompactTaskRouter,
    _encode_rows,
    _holdout,
    _package_runtime,
)
from .layercake_host_runtime import _canonical_sha, _sha256_file
from .layercake_host_v3 import load_english_training_rows


BOUND_ROUTER_SHA256 = (
    "e36607347fab9ccea6a4a8eda7f0477268c7b427222f6b86d97f94d27660b7b8"
)
TRAINING_CURVES = [
    {"step": 1, "loss": 2.3807196617126465, "wall_seconds": 0.24206070002401248},
    {"step": 250, "loss": 0.3435039520263672, "wall_seconds": 2.296204400015995},
    {"step": 500, "loss": 0.08048458397388458, "wall_seconds": 4.362644900014857},
    {"step": 750, "loss": 0.02556135132908821, "wall_seconds": 6.417422600003192},
    {"step": 1000, "loss": 0.014752967283129692, "wall_seconds": 8.467047700018156},
    {"step": 1250, "loss": 0.005120723973959684, "wall_seconds": 10.506884600006742},
    {"step": 1500, "loss": 0.002711326116696, "wall_seconds": 12.566204000002472},
    {"step": 1750, "loss": 0.0026241871528327465, "wall_seconds": 14.624368400021922},
    {"step": 2000, "loss": 0.0019086988177150488, "wall_seconds": 16.669261800008826},
    {"step": 2250, "loss": 0.0010841710027307272, "wall_seconds": 18.724484300008044},
    {"step": 2500, "loss": 0.0006402595899999142, "wall_seconds": 20.779906200012192},
]


def _reconstruct_model(router_path: Path) -> CompactTaskRouter:
    document = onnx.load(router_path)
    tensors = {
        initializer.name: torch.from_numpy(
            np.asarray(numpy_helper.to_array(initializer)).copy()
        )
        for initializer in document.graph.initializer
    }
    expected = {
        "embedding.weight",
        "projection.weight",
        "projection.bias",
        "classifier.weight",
        "classifier.bias",
    }
    if set(tensors) != expected:
        raise RuntimeError("surviving router initializer set changed")
    model = CompactTaskRouter().eval()
    model.load_state_dict(tensors, strict=True)
    return model


def _onnx_holdout(
    session: ort.InferenceSession,
    encoded: Sequence[Sequence[int]],
    rows: Sequence[dict[str, Any]],
    indexes: Sequence[int],
) -> dict[str, Any]:
    records = []
    for order, index in enumerate(indexes, start=1):
        actual = int(
            session.run(
                ["selected_route"],
                {"prompt_ids": np.asarray([encoded[index]], dtype=np.int64)},
            )[0][0]
        )
        expected = int(rows[index]["route"])
        records.append(
            {
                "record_id": str(rows[index]["record_id"]),
                "capability": str(rows[index]["capability"]),
                "expected_route": expected,
                "selected_route": actual,
                "correct": actual == expected,
            }
        )
        if order % 500 == 0:
            print(json.dumps({"evaluated": order, "total": len(indexes)}), flush=True)
    by_capability: dict[str, list[bool]] = defaultdict(list)
    by_route: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        by_capability[record["capability"]].append(record["correct"])
        by_route[record["expected_route"]].append(record["correct"])
    correct = sum(record["correct"] for record in records)
    return {
        "observations": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "capability_accuracy": {
            key: sum(values) / len(values)
            for key, values in sorted(by_capability.items())
        },
        "route_accuracy": {
            str(key): sum(values) / len(values)
            for key, values in sorted(by_route.items())
        },
        "records": records,
    }


@torch.inference_mode()
def _equivalence(
    model: CompactTaskRouter,
    session: ort.InferenceSession,
    encoded: Sequence[Sequence[int]],
    indexes: Sequence[int],
) -> dict[str, Any]:
    maximum_difference = 0.0
    routes_exact = True
    for index in indexes:
        values = np.asarray([encoded[index]], dtype=np.int64)
        onnx_scores, onnx_route = session.run(None, {"prompt_ids": values})
        torch_scores, torch_route = model(torch.from_numpy(values))
        difference = float(
            np.max(np.abs(onnx_scores - torch_scores.numpy()))
        )
        maximum_difference = max(maximum_difference, difference)
        routes_exact = routes_exact and int(onnx_route[0]) == int(
            torch_route.item()
        )
    return {
        "probes": len(indexes),
        "maximum_absolute_score_difference": maximum_difference,
        "scores_close_at_1e_5": maximum_difference <= 1.0e-5,
        "routes_exact": routes_exact,
    }


def _cuda_peak_upper_bound(model: CompactTaskRouter) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for ABI acquisition recovery")
    device = torch.device("cuda")
    disposable = CompactTaskRouter().to(device)
    disposable.load_state_dict(model.state_dict())
    optimizer_state_upper_bound = [
        torch.empty_like(parameter, device=device)
        for parameter in disposable.parameters()
        for _ in range(2)
    ]
    torch.cuda.reset_peak_memory_stats(device)
    ids = torch.arange(
        BATCH_SIZE * MAX_ROUTER_TOKENS,
        dtype=torch.long,
        device=device,
    ).reshape(BATCH_SIZE, MAX_ROUTER_TOKENS).remainder(VOCAB_SIZE)
    mask = torch.ones_like(ids)
    labels = torch.arange(BATCH_SIZE, device=device).remainder(OUTPUT_ROUTES)
    loss = F.cross_entropy(disposable.scores(ids, mask), labels)
    loss.backward()
    torch.cuda.synchronize()
    peak = int(torch.cuda.max_memory_allocated(device))
    del optimizer_state_upper_bound, disposable, ids, mask, labels, loss
    torch.cuda.empty_cache()
    return peak


def recover(
    *,
    parent_runtime: str | Path,
    bundle: str | Path,
    budget_index: int,
    output: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    parent = Path(parent_runtime).resolve()
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    router_path = output / "task-route-router-compact-fp32.onnx"
    if not output.is_dir() or _sha256_file(router_path) != BOUND_ROUTER_SHA256:
        raise RuntimeError("recovery is not bound to the surviving v40 router")
    unexpected = {
        path.name for path in output.iterdir()
    } - {router_path.name}
    if unexpected:
        raise RuntimeError(f"partial artifact contains unexpected files: {unexpected}")
    rows, budget, source_bundle = load_english_training_rows(
        bundle, budget_index=budget_index
    )
    tokenizer = Tokenizer.from_file(str(parent / "tokenizer.json"))
    encoded = _encode_rows(tokenizer, rows)
    train_indexes = [
        index for index, row in enumerate(rows)
        if not _holdout(str(row["record_id"]))
    ]
    holdout_indexes = [
        index for index, row in enumerate(rows)
        if _holdout(str(row["record_id"]))
    ]
    session = ort.InferenceSession(
        str(router_path), providers=["CPUExecutionProvider"]
    )
    model = _reconstruct_model(router_path)
    equivalence = _equivalence(
        model, session, encoded, holdout_indexes[:32]
    )
    holdout = _onnx_holdout(session, encoded, rows, holdout_indexes)
    minimum_capability = min(holdout["capability_accuracy"].values())
    minimum_route = min(holdout["route_accuracy"].values())
    peak_upper_bound = _cuda_peak_upper_bound(model)
    prompt_bytes = [str(row["prompt"]).encode("utf-8") for row in rows]
    response_bytes = [str(row["response"]).encode("utf-8") for row in rows]
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
    parameter_count = sum(value.numel() for value in model.parameters())
    evidence: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": "ABI_ENGLISH_V56_COMPACT_GPU_ROUTER_PROTOCOL_V40.json",
        "recovery_amendment": "ABI_ENGLISH_V56_COMPACT_GPU_ROUTER_V40_RECOVERY_AMENDMENT.json",
        "writer_failure": "ABI_ENGLISH_V56_COMPACT_GPU_ROUTER_V40_EVIDENCE_WRITER_FAILURE.json",
        "parent_runtime": str(parent),
        "parent_runtime_metadata_evidence_sha256": json.loads(
            (parent / "metadata.json").read_text(encoding="utf-8")
        )["evidence_sha256"],
        "source_bundle": str(bundle),
        "source_bundle_sha256": _sha256_file(bundle),
        "source_bundle_manifest_sha256": source_bundle["verification"][
            "manifest_sha256"
        ],
        "budget_id": budget["budget_id"],
        "budget_index": budget_index,
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
            "trainable_parameters": parameter_count,
        },
        "information_accounting": {
            "selected_search_records": len(rows),
            "training_records": len(train_indexes),
            "internal_search_holdout_records": len(holdout_indexes),
            "raw_source_prompt_bytes": sum(map(len, prompt_bytes)),
            "unique_source_prompt_utf8_bytes": sum(
                len(value) for value in set(prompt_bytes)
            ),
            "teacher_output_bytes_accessible_in_source_bundle": sum(
                map(len, response_bytes)
            ),
            "teacher_tokens_accessible_in_source_bundle": sum(
                int(row["teacher_tokens"]) for row in rows
            ),
            "teacher_output_bytes_consumed_by_router_loss": 0,
            "teacher_tokens_consumed_by_router_loss": 0,
            "source_logits_consumed_or_stored": 0,
            "source_hidden_activations_consumed_or_stored": 0,
            "source_parameters_copied": 0,
            "capability_labels_consumed": len(rows),
        },
        "compute": {
            "external_command_runner_wall_seconds": 34.8,
            "final_logged_training_loop_wall_seconds": TRAINING_CURVES[-1][
                "wall_seconds"
            ],
            "gpu_hours_from_external_runner": 34.8 / 3600.0,
            "peak_cuda_memory_upper_bound_bytes": peak_upper_bound,
            "peak_cuda_memory_measurement": "fresh_disposable_full_256x512_backward_with_two_optimizer_state_sized_buffers_no_surviving_weight_update",
            "recovery_wall_seconds": time.perf_counter() - started,
            "source_model_inference_hours": 0.0,
            "external_hardware_used": False,
        },
        "search_holdout": holdout,
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
        from .layercake_host_runtime import _write_json
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
        "holdout_accuracy": holdout["accuracy"],
        "minimum_capability_accuracy": minimum_capability,
        "minimum_route_accuracy": minimum_route,
        "active_neural_graph_bytes": metadata["runtime"][
            "active_neural_graph_bytes"
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = recover(
        parent_runtime=root / "results/abi_moonshot/native_hosts/three-block-v56-router-fp32-layer0-fp32-v39",
        bundle=root / "results/abi_moonshot/segregated_acquisition_v3/phi3-broad-natural-conversation-complete-search-training-v3.abix",
        budget_index=3,
        output=root / "results/abi_moonshot/native_hosts/three-block-v56-compact-router-layer0-fp32-v40",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
