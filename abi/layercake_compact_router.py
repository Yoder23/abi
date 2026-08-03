"""GPU acquisition and immutable packaging for one compact LayerCake router."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from typing import Any, Mapping, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from .layercake_host_runtime import (
    COMPACT_TASK_ROUTE_ROUTER_MODE,
    RUNTIME_FORMAT,
    _canonical_sha,
    _sha256_file,
    _write_json,
)
from .layercake_host_v3 import load_english_training_rows


FORMAT = "abi-layercake-compact-task-router-training/1"
SEED = 56040
VOCAB_SIZE = 50_257
EMBEDDING_WIDTH = 64
HIDDEN_WIDTH = 128
OUTPUT_ROUTES = 10
MAX_ROUTER_TOKENS = 512
BATCH_SIZE = 256
STEPS = 2_500
LEARNING_RATE = 3.0e-3
WEIGHT_DECAY = 1.0e-4


class CompactTaskRouter(nn.Module):
    """Small sequence classifier; it does not realize or store English text."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_WIDTH)
        self.projection = nn.Linear(2 * EMBEDDING_WIDTH, HIDDEN_WIDTH)
        self.classifier = nn.Linear(HIDDEN_WIDTH, OUTPUT_ROUTES)

    def scores(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids[:, :MAX_ROUTER_TOKENS]
        attention_mask = attention_mask[:, :MAX_ROUTER_TOKENS]
        hidden = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        mean = (hidden * mask).sum(dim=1) / denominator
        masked = hidden.masked_fill(mask == 0, torch.finfo(hidden.dtype).min)
        maximum = masked.max(dim=1).values
        pooled = torch.cat((mean, maximum), dim=-1)
        return self.classifier(F.gelu(self.projection(pooled)))

    def forward(
        self, prompt_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_ids = prompt_ids[:, :MAX_ROUTER_TOKENS]
        attention_mask = torch.ones_like(prompt_ids)
        scores = self.scores(prompt_ids, attention_mask)
        return scores, scores.argmax(dim=-1)


def _holdout(record_id: str) -> bool:
    return hashlib.sha256(record_id.encode("utf-8")).digest()[0] < 26


def _encode_rows(
    tokenizer: Tokenizer,
    rows: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
    encodings = tokenizer.encode_batch(
        [str(row["prompt"]) for row in rows]
    )
    return [
        encoding.ids[:MAX_ROUTER_TOKENS] or [50_256]
        for encoding in encodings
    ]


def _batch(
    encoded: Sequence[Sequence[int]],
    routes: Sequence[int],
    indexes: Sequence[int],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(encoded[index]) for index in indexes)
    ids = torch.full(
        (len(indexes), width),
        50_256,
        dtype=torch.long,
    )
    mask = torch.zeros((len(indexes), width), dtype=torch.long)
    labels = torch.empty(len(indexes), dtype=torch.long)
    for row_index, source_index in enumerate(indexes):
        values = encoded[source_index]
        ids[row_index, : len(values)] = torch.tensor(values)
        mask[row_index, : len(values)] = 1
        labels[row_index] = int(routes[source_index])
    return (
        ids.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
    )


@torch.inference_mode()
def _evaluate(
    model: CompactTaskRouter,
    encoded: Sequence[Sequence[int]],
    rows: Sequence[Mapping[str, Any]],
    indexes: Sequence[int],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(indexes), BATCH_SIZE):
        selected = list(indexes[start : start + BATCH_SIZE])
        ids, mask, labels = _batch(
            encoded,
            [int(row["route"]) for row in rows],
            selected,
            device=device,
        )
        predicted = model.scores(ids, mask).argmax(dim=-1)
        for source_index, expected, actual in zip(
            selected,
            labels.tolist(),
            predicted.tolist(),
            strict=True,
        ):
            records.append(
                {
                    "record_id": str(rows[source_index]["record_id"]),
                    "capability": str(rows[source_index]["capability"]),
                    "expected_route": int(expected),
                    "selected_route": int(actual),
                    "correct": bool(expected == actual),
                }
            )
    by_capability: dict[str, list[bool]] = {}
    by_route: dict[int, list[bool]] = {}
    for record in records:
        by_capability.setdefault(record["capability"], []).append(
            record["correct"]
        )
        by_route.setdefault(record["expected_route"], []).append(
            record["correct"]
        )
    return {
        "observations": len(records),
        "correct": sum(record["correct"] for record in records),
        "accuracy": sum(record["correct"] for record in records)
        / len(records),
        "capability_accuracy": {
            capability: sum(values) / len(values)
            for capability, values in sorted(by_capability.items())
        },
        "route_accuracy": {
            str(route): sum(values) / len(values)
            for route, values in sorted(by_route.items())
        },
        "records": records,
    }


def _sample_balanced_capabilities(
    indexes_by_capability: Mapping[str, Sequence[int]],
    *,
    rng: random.Random,
) -> list[int]:
    capabilities = sorted(indexes_by_capability)
    selected = []
    for offset in range(BATCH_SIZE):
        capability = capabilities[offset % len(capabilities)]
        selected.append(rng.choice(indexes_by_capability[capability]))
    rng.shuffle(selected)
    return selected


def _export_router(
    model: CompactTaskRouter,
    path: Path,
) -> None:
    cpu_model = CompactTaskRouter().eval()
    cpu_model.load_state_dict(
        {name: value.detach().cpu() for name, value in model.state_dict().items()}
    )
    torch.onnx.export(
        cpu_model,
        (torch.tensor([[464, 318, 257, 1332]], dtype=torch.long),),
        path,
        input_names=["prompt_ids"],
        output_names=["task_scores", "selected_route"],
        dynamic_axes={
            "prompt_ids": {1: "prompt_sequence"},
            "task_scores": {0: "batch"},
            "selected_route": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(path))


def _package_runtime(
    *,
    parent: Path,
    output: Path,
    router_path: Path,
    training_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    parent_metadata = json.loads(
        (parent / "metadata.json").read_text(encoding="utf-8")
    )
    if (
        parent_metadata.get("format") != RUNTIME_FORMAT
        or parent_metadata.get("status")
        not in {"PASS", "EXPORTED_NOT_YET_CERTIFIED"}
        or parent_metadata.get("evidence_sha256")
        != "d8f295a00cb900960652481d123d966d0789ed6c744283a0f78463ec6d245e8a"
    ):
        raise RuntimeError("compact-router parent runtime is not qualified")
    runtime = parent_metadata["runtime"]
    control = runtime["layerwise_capability_control"]
    if (
        control.get("router_mode")
        != "onnx_zero_control_transformer_mean_classifier"
        or control.get("router_precision") != "fp32"
    ):
        raise RuntimeError("compact-router parent is not the locked v39 host")
    graph_source = parent / str(runtime["graph"])
    tokenizer_source = parent / str(parent_metadata["tokenizer"]["path"])
    symbolic_source = parent / str(parent_metadata["symbolic_surface"]["path"])
    for source in (graph_source, tokenizer_source, symbolic_source):
        if not source.is_file():
            raise RuntimeError(f"runtime component is absent: {source.name}")
    shutil.copy2(graph_source, output / graph_source.name)
    shutil.copy2(tokenizer_source, output / tokenizer_source.name)
    shutil.copy2(symbolic_source, output / symbolic_source.name)
    final_router = output / "task-route-router-compact-fp32.onnx"
    if router_path != final_router:
        shutil.copy2(router_path, final_router)
    evidence_path = output / "compact-router-training-evidence.json"
    _write_json(evidence_path, training_evidence)

    metadata = json.loads(json.dumps(parent_metadata))
    metadata.pop("evidence_sha256", None)
    compact = metadata["runtime"]["layerwise_capability_control"]
    compact.update(
        {
            "router_graph": final_router.name,
            "router_graph_sha256": _sha256_file(final_router),
            "router_graph_bytes": final_router.stat().st_size,
            "router_mode": COMPACT_TASK_ROUTE_ROUTER_MODE,
            "router_precision": "fp32",
            "router_provider": "onnxruntime.CPUExecutionProvider",
            "routing_prepass_uses_zero_control": False,
            "separate_router_session": True,
            "training_device": "cuda",
            "maximum_router_tokens": MAX_ROUTER_TOKENS,
            "embedding_width": EMBEDDING_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "output_routes": OUTPUT_ROUTES,
            "training_evidence": evidence_path.name,
            "training_evidence_sha256": _sha256_file(evidence_path),
        }
    )
    metadata["runtime"]["active_neural_graph_bytes"] = (
        int(metadata["runtime"]["graph_bytes"])
        + int(compact["router_graph_bytes"])
    )
    metadata["compact_router_acquisition"] = {
        "format": FORMAT,
        "parent_runtime_metadata_evidence_sha256": parent_metadata[
            "evidence_sha256"
        ],
        "training_evidence": evidence_path.name,
        "training_evidence_sha256": _sha256_file(evidence_path),
        "training_device": "cuda",
        "teacher_present_at_inference": False,
        "source_teacher_text_retained": False,
        "source_logits_retained": 0,
        "source_hidden_activations_retained": 0,
        "source_parameters_copied": 0,
        "english_checkpoint_changed": False,
        "task_cakes_changed": False,
        "layerwise_controls_changed": False,
        "symbolic_substrate_changed": False,
    }
    metadata["evidence_sha256"] = _canonical_sha(metadata)
    _write_json(output / "metadata.json", metadata)
    return metadata


def train_and_package(
    *,
    parent_runtime: str | Path,
    bundle: str | Path,
    budget_index: int,
    output: str | Path,
) -> dict[str, Any]:
    parent = Path(parent_runtime).resolve()
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise RuntimeError(f"compact-router artifact is immutable: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for ABI router acquisition")
    output.mkdir(parents=True)
    rows, budget, source_bundle = load_english_training_rows(
        bundle, budget_index=budget_index
    )
    tokenizer_path = parent / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encoded = _encode_rows(tokenizer, rows)
    train_indexes = [
        index
        for index, row in enumerate(rows)
        if not _holdout(str(row["record_id"]))
    ]
    holdout_indexes = [
        index
        for index, row in enumerate(rows)
        if _holdout(str(row["record_id"]))
    ]
    indexes_by_capability: dict[str, list[int]] = {}
    for index in train_indexes:
        indexes_by_capability.setdefault(
            str(rows[index]["capability"]), []
        ).append(index)
    if len(indexes_by_capability) != 14 or not holdout_indexes:
        raise RuntimeError("router search split lost required capabilities")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng = random.Random(SEED)
    device = torch.device("cuda")
    model = CompactTaskRouter().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    process_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    sampled_prompt_tokens = 0
    curves = []
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
            curves.append(
                {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "wall_seconds": time.perf_counter() - process_started,
                }
            )
            print(json.dumps(curves[-1]), flush=True)
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - process_started
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
    pytorch_routes = []
    model.eval()
    with torch.inference_mode():
        for index in probe_indexes:
            ids = torch.tensor([encoded[index]], dtype=torch.long, device=device)
            mask = torch.ones_like(ids)
            pytorch_routes.append(
                int(model.scores(ids, mask).argmax(dim=-1).item())
            )
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
    parameter_count = sum(value.numel() for value in model.parameters())
    prompt_bytes = [str(row["prompt"]).encode("utf-8") for row in rows]
    response_bytes = [str(row["response"]).encode("utf-8") for row in rows]
    gates = {
        "overall_route_accuracy_at_least_095": holdout["accuracy"] >= 0.95,
        "every_capability_route_accuracy_at_least_080": minimum_capability
        >= 0.80,
        "every_represented_route_accuracy_at_least_080": minimum_route >= 0.80,
        "onnx_matches_pytorch_on_bound_probes": onnx_matches,
        "training_device_is_cuda": device.type == "cuda",
    }
    evidence: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol": "ABI_ENGLISH_V56_COMPACT_GPU_ROUTER_PROTOCOL_V40.json",
        "parent_runtime": str(parent),
        "parent_runtime_metadata_evidence_sha256": json.loads(
            (parent / "metadata.json").read_text(encoding="utf-8")
        )["evidence_sha256"],
        "source_bundle": str(bundle),
        "source_bundle_sha256": _sha256_file(bundle),
        "source_bundle_manifest_sha256": source_bundle["manifest"][
            "manifest_sha256"
        ],
        "budget_id": budget["budget_id"],
        "budget_index": budget_index,
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
        "search_holdout": holdout,
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
        "holdout_accuracy": holdout["accuracy"],
        "minimum_capability_accuracy": minimum_capability,
        "minimum_route_accuracy": minimum_route,
        "router_graph_bytes": router_path.stat().st_size,
        "active_neural_graph_bytes": metadata["runtime"][
            "active_neural_graph_bytes"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-runtime", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--budget-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_and_package(
        parent_runtime=args.parent_runtime,
        bundle=args.bundle,
        budget_index=args.budget_index,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
