"""One preregistered GPU router run with non-overlapping rare-route data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
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
    STEPS,
    VOCAB_SIZE,
    WEIGHT_DECAY,
    CompactTaskRouter,
    _batch,
    _encode_rows,
    _evaluate,
    _export_router,
    _holdout,
    _package_runtime,
    _sample_balanced_capabilities,
)
from .layercake_host_runtime import _canonical_sha, _sha256_file, _write_json
from .layercake_host_v3 import load_english_training_rows


SEED = 56041


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _fresh_holdout(prompt: str) -> bool:
    return hashlib.sha256(prompt.encode("utf-8")).digest()[0] < 26


def _deduplicate_v9(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_prompt_sha(str(row["prompt"])), []).append(row)
    selected = []
    for prompt_sha, values in sorted(grouped.items()):
        routes = {int(row["route"]) for row in values}
        if len(routes) != 1:
            raise RuntimeError(
                f"v9 duplicate prompt has conflicting routes: {prompt_sha}"
            )
        selected.append(
            dict(min(values, key=lambda row: str(row["record_id"])))
        )
    return selected


def train_and_package(
    *,
    parent_runtime: str | Path,
    original_bundle: str | Path,
    v9_bundle: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    parent = Path(parent_runtime).resolve()
    original_bundle = Path(original_bundle).resolve()
    v9_bundle = Path(v9_bundle).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError(f"balanced-router artifact is immutable: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for ABI router acquisition")
    output.mkdir(parents=True)
    original_rows, original_budget, original_source = (
        load_english_training_rows(original_bundle, budget_index=3)
    )
    v9_rows_raw, v9_budget, v9_source = load_english_training_rows(
        v9_bundle, budget_index=-1
    )
    v9_rows = _deduplicate_v9(v9_rows_raw)
    original_training = [
        dict(row)
        for row in original_rows
        if not _holdout(str(row["record_id"]))
    ]
    v9_training = [
        row for row in v9_rows if not _fresh_holdout(str(row["prompt"]))
    ]
    v9_holdout = [
        row for row in v9_rows if _fresh_holdout(str(row["prompt"]))
    ]
    rows = [*original_training, *v9_training, *v9_holdout]
    training_count = len(original_training) + len(v9_training)
    training_indexes = list(range(training_count))
    holdout_indexes = list(range(training_count, len(rows)))
    tokenizer = Tokenizer.from_file(str(parent / "tokenizer.json"))
    encoded = _encode_rows(tokenizer, rows)
    indexes_by_capability: dict[str, list[int]] = {}
    for index in training_indexes:
        indexes_by_capability.setdefault(
            str(rows[index]["capability"]), []
        ).append(index)
    if len(indexes_by_capability) != 14:
        raise RuntimeError("balanced router lost one required capability")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED)
    device = torch.device("cuda")
    model = CompactTaskRouter().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    curves = []
    sampled_prompt_tokens = 0
    routes = [int(row["route"]) for row in rows]
    model.train()
    for step in range(1, STEPS + 1):
        indexes = _sample_balanced_capabilities(
            indexes_by_capability, rng=rng
        )
        ids, mask, labels = _batch(
            encoded, routes, indexes, device=device
        )
        sampled_prompt_tokens += int(mask.sum().item())
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model.scores(ids, mask), labels)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 250 == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(row)
            print(json.dumps(row), flush=True)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - started
    holdout = _evaluate(
        model,
        encoded,
        rows,
        holdout_indexes,
        device=device,
    )
    minimum_capability = min(holdout["capability_accuracy"].values())
    minimum_route = min(holdout["route_accuracy"].values())

    router_path = output / "task-route-router-compact-fp32.onnx"
    _export_router(model, router_path)
    probe_indexes = holdout_indexes[:32]
    model.eval()
    with torch.inference_mode():
        pytorch_routes = [
            int(
                model.scores(
                    torch.tensor([encoded[index]], dtype=torch.long, device=device),
                    torch.ones(
                        (1, len(encoded[index])), dtype=torch.long, device=device
                    ),
                )
                .argmax(dim=-1)
                .item()
            )
            for index in probe_indexes
        ]
    session = ort.InferenceSession(
        str(router_path), providers=["CPUExecutionProvider"]
    )
    onnx_routes = [
        int(
            session.run(
                ["selected_route"],
                {"prompt_ids": np.asarray([encoded[index]], dtype=np.int64)},
            )[0][0]
        )
        for index in probe_indexes
    ]
    onnx_matches = onnx_routes == pytorch_routes
    gates = {
        "overall_route_accuracy_at_least_095": holdout["accuracy"] >= 0.95,
        "every_capability_route_accuracy_at_least_080": minimum_capability >= 0.80,
        "every_represented_route_accuracy_at_least_080": minimum_route >= 0.80,
        "onnx_matches_pytorch_on_bound_probes": onnx_matches,
        "training_device_is_cuda": True,
    }
    source_rows = [*original_rows, *v9_rows_raw]
    prompt_bytes = [str(row["prompt"]).encode("utf-8") for row in source_rows]
    response_bytes = [str(row["response"]).encode("utf-8") for row in source_rows]
    evidence: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": "ABI_ENGLISH_V56_BALANCED_COMPACT_GPU_ROUTER_PROTOCOL_V41.json",
        "parent_runtime": str(parent),
        "parent_runtime_metadata_evidence_sha256": json.loads(
            (parent / "metadata.json").read_text(encoding="utf-8")
        )["evidence_sha256"],
        "source_bundles": [
            {
                "path": str(original_bundle),
                "sha256": _sha256_file(original_bundle),
                "manifest_sha256": original_source["verification"][
                    "manifest_sha256"
                ],
                "budget_id": original_budget["budget_id"],
                "budget_index": 3,
            },
            {
                "path": str(v9_bundle),
                "sha256": _sha256_file(v9_bundle),
                "manifest_sha256": v9_source["verification"][
                    "manifest_sha256"
                ],
                "budget_id": v9_budget["budget_id"],
                "budget_index": -1,
            },
        ],
        "device": "cuda",
        "cuda_device_name": torch.cuda.get_device_name(device),
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
            "trainable_parameters": sum(
                value.numel() for value in model.parameters()
            ),
            "changes_vs_v40": 0,
        },
        "information_accounting": {
            "source_bundle_records": len(source_rows),
            "original_training_records_after_v40_holdout_exclusion": len(
                original_training
            ),
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
            "sampled_prompt_tokens_seen": sampled_prompt_tokens,
        },
        "compute": {
            "wall_seconds": wall_seconds,
            "gpu_hours": wall_seconds / 3600.0,
            "peak_cuda_memory_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "source_model_inference_hours": 0.0,
            "external_hardware_used": False,
        },
        "fresh_v9_holdout": holdout,
        "minimum_capability_accuracy": minimum_capability,
        "minimum_route_accuracy": minimum_route,
        "onnx_equivalence_probe_record_ids": [
            str(rows[index]["record_id"]) for index in probe_indexes
        ],
        "onnx_matches_pytorch_on_bound_probes": onnx_matches,
        "router_graph_sha256": _sha256_file(router_path),
        "router_graph_bytes": router_path.stat().st_size,
        "curves": curves,
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
        "active_neural_graph_bytes": metadata["runtime"][
            "active_neural_graph_bytes"
        ],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    result = train_and_package(
        parent_runtime=root / "results/abi_moonshot/native_hosts/three-block-v56-router-fp32-layer0-fp32-v39",
        original_bundle=root / "results/abi_moonshot/segregated_acquisition_v3/phi3-broad-natural-conversation-complete-search-training-v3.abix",
        v9_bundle=root / "results/abi_moonshot/segregated_acquisition_v3/phi3-broad-cumulative-english-training-v9.abix",
        output=root / "results/abi_moonshot/native_hosts/three-block-v56-balanced-compact-router-layer0-fp32-v41",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
