"""Train and verify the preregistered segment-aware ABI capability router."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import psutil
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn

from .capability_compiler_phase2_common import (
    CAPABILITIES,
    canonical_json_bytes,
    set_determinism,
    sha256_file,
)
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import (
    Phase3Error,
    _BalancedSampler,
    _write_immutable,
    load_phase1_ir,
)
from .capability_compiler_phase3_bpe_core import _json, _layercake_api, _tokenizer


FORMAT = "abi-capability-compiler-phase3-segment-router/1"
METADATA = "__metadata__"


class SegmentRouter(nn.Module):
    """Small order-sensitive segment classifier with an explicit metadata class."""

    def __init__(
        self,
        vocabulary: int,
        embedding_width: int,
        channels: int,
        hidden_width: int,
        kernels: Sequence[int],
        classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.padding_id = vocabulary
        self.kernels = tuple(int(value) for value in kernels)
        self.embedding = nn.Embedding(
            vocabulary + 1, embedding_width, padding_idx=self.padding_id
        )
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_width, channels, kernel_size=kernel)
            for kernel in self.kernels
        )
        combined = channels * len(self.kernels)
        self.norm = nn.LayerNorm(combined)
        self.hidden = nn.Linear(combined, hidden_width)
        self.output = nn.Linear(hidden_width, classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.embedding(ids).transpose(1, 2)
        pooled = []
        for kernel, convolution in zip(self.kernels, self.convolutions):
            features = F.gelu(convolution(encoded))
            valid = torch.clamp(lengths - kernel + 1, min=1)
            positions = torch.arange(features.shape[-1], device=features.device)[None, :]
            features = features.masked_fill(positions[:, None, :] >= valid[:, None, None], -torch.inf)
            pooled.append(features.amax(dim=-1))
        joined = self.norm(torch.cat(pooled, dim=-1))
        return self.output(self.dropout(F.gelu(self.hidden(joined))))


def load_protocol(root: Path, path: Path) -> tuple[Mapping[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SEGMENT_ROUTER_GATE"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("training", {}).get("device") != "cuda"
    ):
        raise Phase3Error("V44 governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V44 binding changed: {relative}")
    return protocol, sha256_file(path)


def _data(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for row in load_phase1_ir((root / protocol["phase1_ir"]).resolve()):
        prompt = str(row["normalized_acquisition_prompt"])
        lines = prompt.splitlines()
        if len(lines) < 2 or not lines[0].strip():
            raise Phase3Error("acquisition record lacks a metadata/body boundary")
        body = "\n".join(lines[1:]).strip()
        rows.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "metadata": lines[0].strip(),
                "body": body,
                "metadata_ids": _encode(tokenizer, lines[0].strip()),
                "body_ids": _encode(tokenizer, body),
            }
        )
    return rows


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = [tokenizer.lexeme_to_id[item] for item in tokenizer.split(text)]
    if not values:
        raise Phase3Error("empty segment")
    return values


def _collate(
    sequences: Sequence[Sequence[int]], padding_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long, device=device)
    width = max(max(int(lengths.max()), 1), 5)
    ids = torch.full((len(sequences), width), padding_id, dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        ids[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return ids, lengths


def _model(protocol: Mapping[str, Any], vocabulary: int) -> SegmentRouter:
    architecture = protocol["architecture"]
    return SegmentRouter(
        vocabulary,
        int(architecture["embedding_width"]),
        int(architecture["channels"]),
        int(architecture["hidden_width"]),
        tuple(architecture["kernels"]),
        len(CAPABILITIES) + 1,
        float(architecture["dropout"]),
    )


def inventory(root: Path, path: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    rows = _data(root, protocol, tokenizer)
    model = _model(protocol, tokenizer.vocab_size)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error(f"segment-router parameter count changed: {parameters}")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "records": len(rows),
        "capabilities": len(CAPABILITIES),
        "unique_metadata_segments": len({row["metadata"] for row in rows}),
        "trainable_parameters": parameters,
        "maximum_body_actions": max(len(row["body_ids"]) for row in rows),
        "maximum_metadata_actions": max(len(row["metadata_ids"]) for row in rows),
        "teacher_outputs_added": 0,
        "final_test_accessed": False,
    }


def train(root: Path, path: Path, output: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("segment-router output exists or CUDA unavailable")
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    rows = _data(root, protocol, tokenizer)
    config = protocol["training"]
    seed = int(config["seed"])
    set_determinism(seed)
    device = torch.device("cuda")
    model = _model(protocol, tokenizer.vocab_size).to(device)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(config["trainable_parameters"]):
        raise Phase3Error("segment-router parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    sampler = _BalancedSampler(rows, seed)
    label_to_id = {name: index for index, name in enumerate((*CAPABILITIES, METADATA))}
    class_weights = torch.ones(len(label_to_id), device=device)
    class_weights[label_to_id[METADATA]] = 1.0 / len(CAPABILITIES)
    curves = []
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    sequence_hash = hashlib.sha256()
    model.train()
    for step in range(1, int(config["steps"]) + 1):
        batch = sampler.batch(int(config["batch_size"]))
        sequences: list[list[int]] = []
        targets: list[int] = []
        for row in batch:
            sequences.extend((row["body_ids"], row["metadata_ids"]))
            targets.extend((label_to_id[row["capability"]], label_to_id[METADATA]))
            sequence_hash.update(row["record_id"].encode() + b"\n")
        ids, lengths = _collate(sequences, model.padding_id, device)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, lengths)
        loss = F.cross_entropy(logits, target, weight=class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(config["curve_interval"]) == 0:
            predictions = logits.argmax(dim=-1)
            body_positions = torch.arange(len(targets), device=device) % 2 == 0
            value = {
                "step": step,
                "loss": float(loss.detach()),
                "body_accuracy": float(predictions[body_positions].eq(target[body_positions]).float().mean()),
                "metadata_accuracy": float(predictions[~body_positions].eq(target[~body_positions]).float().mean()),
                "wall_seconds": time.perf_counter() - started,
            }
            curves.append(value)
            print(json.dumps(value), flush=True)
    output.mkdir(parents=True)
    checkpoint = output / "router.safetensors"
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()},
        str(checkpoint),
    )
    config_path = output / "config.json"
    _write_immutable(
        config_path,
        json.dumps(
            {"vocabulary": tokenizer.vocab_size, **protocol["architecture"]},
            sort_keys=True,
            indent=2,
        ).encode()
        + b"\n",
    )
    metadata: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-segment-router-candidate/1",
        "status": "TRAINED_SEGMENT_ROUTER_GATE_ONLY",
        "protocol_sha256": protocol_hash,
        "seed": seed,
        "checkpoint": {
            "path": "router.safetensors",
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "config": {
            "path": "config.json",
            "sha256": sha256_file(config_path),
            "trainable_parameters": parameters,
        },
        "training": {
            "steps": int(config["steps"]),
            "batch_size": int(config["batch_size"]),
            "effective_segments_per_step": 2 * int(config["batch_size"]),
            "wall_seconds": time.perf_counter() - started,
            "record_sequence_sha256": sequence_hash.hexdigest(),
            "peak_process_rss_bytes": peak_rss,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "curves": curves,
        },
        "imported_information": {
            "records": 7000,
            "capability_labels": 7000,
            "metadata_labels_derived_from_existing_prompt_boundaries": 7000,
            "teacher_outputs_added": 0,
            "stored_logits": 0,
            "stored_activations": 0,
            "source_parameters_copied": 0,
        },
        "teacher_present_at_inference": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "hardware": {"machine": platform.node(), "gpu": torch.cuda.get_device_name(0)},
    }
    metadata["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(metadata)).hexdigest()
    _write_immutable(
        output / "metadata.json",
        json.dumps(metadata, sort_keys=True, indent=2).encode() + b"\n",
    )
    return metadata


def _load(
    root: Path, protocol: Mapping[str, Any], candidate: Path
) -> tuple[SegmentRouter, Any]:
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    model = _model(protocol, tokenizer.vocab_size)
    model.load_state_dict(
        load_file(str(candidate / "router.safetensors"), device="cuda"), strict=True
    )
    return model.cuda().eval(), tokenizer


def _semantic_segments(text: str) -> list[str]:
    lines = text.splitlines()
    if len(lines) < 2:
        return [text.strip()]
    first = lines[0].strip()
    remainder = "\n".join(lines[1:]).strip()
    return [part for part in (first, remainder) if part]


@torch.inference_mode()
def _route(
    model: SegmentRouter, tokenizer: Any, text: str
) -> tuple[str, list[dict[str, Any]]]:
    segments = _semantic_segments(text)
    encoded = [_encode(tokenizer, segment) for segment in segments]
    ids, lengths = _collate(encoded, model.padding_id, torch.device("cuda"))
    probabilities = model(ids, lengths).softmax(dim=-1).cpu()
    labels = (*CAPABILITIES, METADATA)
    details = []
    for segment, row in zip(segments, probabilities):
        predicted = int(row.argmax())
        capability_probability, capability_index = row[: len(CAPABILITIES)].max(dim=0)
        details.append(
            {
                "segment_sha256": hashlib.sha256(segment.encode()).hexdigest(),
                "predicted": labels[predicted],
                "predicted_probability": float(row[predicted]),
                "best_capability": CAPABILITIES[int(capability_index)],
                "best_capability_probability": float(capability_probability),
            }
        )
    eligible = [item for item in details if item["predicted"] != METADATA]
    selected = max(
        eligible if eligible else details,
        key=lambda item: item["best_capability_probability"],
    )
    return str(selected["best_capability"]), details


@torch.inference_mode()
def evaluate(root: Path, path: Path, candidate: Path, output: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    metadata = _json(candidate / "metadata.json")
    if (
        output.exists()
        or metadata.get("protocol_sha256") != protocol_hash
        or sha256_file(candidate / "router.safetensors") != metadata["checkpoint"]["sha256"]
    ):
        raise Phase3Error("segment-router evaluation identity failed")
    model, tokenizer = _load(root, protocol, candidate)
    probes = development_probes((root / protocol["development_catalog"]).resolve())
    rows = []
    for probe in probes:
        capability = str(probe["canonical_capability"])
        prompt = str(probe["prompt"])
        lines = prompt.splitlines()
        body = "\n".join(lines[1:]).strip()
        metadata_segment = lines[0].strip()
        for variant, text in (("original", prompt), ("body", body)):
            prediction, details = _route(model, tokenizer, text)
            rows.append(
                {
                    "probe_id": str(probe["probe_id"]),
                    "capability": capability,
                    "variant": variant,
                    "predicted": prediction,
                    "correct": prediction == capability,
                    "segments": details,
                }
            )
        metadata_ids, metadata_lengths = _collate(
            [_encode(tokenizer, metadata_segment)], model.padding_id, torch.device("cuda")
        )
        metadata_prediction = int(model(metadata_ids, metadata_lengths).argmax(dim=-1)[0])
        rows.append(
            {
                "probe_id": str(probe["probe_id"]),
                "capability": capability,
                "variant": "metadata",
                "predicted": (*CAPABILITIES, METADATA)[metadata_prediction],
                "correct": metadata_prediction == len(CAPABILITIES),
            }
        )
    output.mkdir(parents=True)
    raw_path = output / "rows.jsonl"
    raw_path.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    decision = _decision(
        protocol, protocol_hash, metadata, rows, sha256_file(raw_path)
    )
    _write_immutable(
        output / "decision.json",
        json.dumps(decision, sort_keys=True, indent=2).encode() + b"\n",
    )
    return decision


def _wilson(correct: int, observations: int, z: float = 1.959963984540054) -> Mapping[str, float]:
    point = correct / observations
    denominator = 1 + z * z / observations
    center = (point + z * z / (2 * observations)) / denominator
    half = z * math.sqrt(
        point * (1 - point) / observations + z * z / (4 * observations * observations)
    ) / denominator
    return {"point": point, "lower_95": center - half, "upper_95": center + half}


def _decision(
    protocol: Mapping[str, Any],
    protocol_hash: str,
    metadata: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    raw_hash: str,
) -> Mapping[str, Any]:
    summaries = {}
    for variant in ("original", "body", "metadata"):
        subset = [row for row in rows if row["variant"] == variant]
        correct = sum(bool(row["correct"]) for row in subset)
        summary: dict[str, Any] = {
            "correct": correct,
            "observations": len(subset),
            "wilson": _wilson(correct, len(subset)),
            "predicted_counts": dict(sorted(Counter(row["predicted"] for row in subset).items())),
        }
        if variant != "metadata":
            summary["per_capability"] = {
                capability: _wilson(
                    sum(
                        bool(row["correct"])
                        for row in subset
                        if row["capability"] == capability
                    ),
                    100,
                )
                for capability in CAPABILITIES
            }
        summaries[variant] = summary
    gate = protocol["router_gate"]
    original = summaries["original"]
    body = summaries["body"]
    metadata_summary = summaries["metadata"]
    aggregate_original = (
        original["wilson"]["point"] >= float(gate["aggregate_point_minimum"])
        and original["wilson"]["lower_95"]
        >= float(gate["aggregate_wilson_lower_minimum"])
    )
    aggregate_body = body["wilson"]["point"] >= float(gate["body_point_minimum"])
    per_capability = all(
        value["point"] >= float(gate["per_capability_point_minimum"])
        and value["lower_95"] >= float(gate["per_capability_wilson_lower_minimum"])
        for value in original["per_capability"].values()
    )
    metadata_rejection = (
        metadata_summary["wilson"]["point"] >= float(gate["metadata_point_minimum"])
        and metadata_summary["wilson"]["lower_95"]
        >= float(gate["metadata_wilson_lower_minimum"])
    )
    passed = aggregate_original and aggregate_body and per_capability and metadata_rejection
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-segment-router-decision/1",
        "status": (
            "PASS_SEGMENT_ROUTER_GATE_REPLICATION_OPEN"
            if passed
            else "FAIL_SEGMENT_ROUTER_GATE_ARCHITECTURE_CLOSED"
        ),
        "protocol": {
            "path": "ABI_CAPABILITY_COMPILER_PHASE3_SEGMENT_ROUTER_PROTOCOL_V44.json",
            "sha256": protocol_hash,
        },
        "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "summaries": summaries,
        "gates": {
            "aggregate_original": aggregate_original,
            "aggregate_body": aggregate_body,
            "per_capability_original": per_capability,
            "metadata_rejection": metadata_rejection,
            "router_gate_pass": passed,
        },
        "rows_sha256": raw_hash,
        "teacher_outputs_added": 0,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
        "next_step": (
            "Preregister the two locked replication seeds only; a host remains separately gated."
            if passed
            else "Preserve failure; replications and routed host remain prohibited."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train", "evaluate"))
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_SEGMENT_ROUTER_PROTOCOL_V44.json",
    )
    parser.add_argument(
        "--candidate-dir",
        default="results/abi_capability_compiler_phase3_segment_router/development_v44/R0-seed240044",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_segment_router/evaluation_v44/R0-seed240044",
    )
    arguments = parser.parse_args(argv)
    root = Path.cwd().resolve()
    path = (root / arguments.protocol).resolve()
    if arguments.command == "inventory":
        result = inventory(root, path)
    elif arguments.command == "train":
        result = train(root, path, (root / arguments.candidate_dir).resolve())
    else:
        result = evaluate(
            root,
            path,
            (root / arguments.candidate_dir).resolve(),
            (root / arguments.output_dir).resolve(),
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
