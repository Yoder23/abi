"""Sparse multi-granular segment router for the bounded Phase 3 V45 screen."""
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
from .capability_compiler_phase3_segment_router import METADATA, _semantic_segments


FORMAT = "abi-capability-compiler-phase3-sparse-router/1"


def _fnv1a(data: bytes, seed: int) -> int:
    value = (2166136261 ^ seed) & 0xFFFFFFFF
    for byte in data:
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _character_features(
    text: str, buckets: int, minimum: int, maximum: int, seed: int
) -> list[int]:
    normalized = " ".join(text.casefold().split())
    bounded = "^" + normalized + "$"
    features = []
    for width in range(minimum, maximum + 1):
        for start in range(max(0, len(bounded) - width + 1)):
            gram = bounded[start : start + width].encode("utf-8")
            features.append(_fnv1a(gram, seed) % buckets)
    if not features:
        features.append(_fnv1a(bounded.encode("utf-8"), seed) % buckets)
    return features


class SparseRouter(nn.Module):
    """A log-linear sum of BPE unigram and hashed character-phrase evidence."""

    def __init__(self, vocabulary: int, buckets: int, classes: int) -> None:
        super().__init__()
        self.bpe = nn.EmbeddingBag(vocabulary, classes, mode="mean")
        self.character = nn.EmbeddingBag(buckets, classes, mode="mean")
        self.bias = nn.Parameter(torch.zeros(classes))

    def forward(
        self,
        bpe_ids: torch.Tensor,
        bpe_offsets: torch.Tensor,
        character_ids: torch.Tensor,
        character_offsets: torch.Tensor,
    ) -> torch.Tensor:
        return self.bpe(bpe_ids, bpe_offsets) + self.character(
            character_ids, character_offsets
        ) + self.bias


def load_protocol(root: Path, path: Path) -> tuple[Mapping[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SPARSE_ROUTER_GATE"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("training", {}).get("device") != "cuda"
    ):
        raise Phase3Error("V45 governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"V45 binding changed: {relative}")
    return protocol, sha256_file(path)


def _encode(tokenizer: Any, text: str) -> list[int]:
    values = [tokenizer.lexeme_to_id[item] for item in tokenizer.split(text)]
    if not values:
        raise Phase3Error("empty BPE segment")
    return values


def _features(
    tokenizer: Any, protocol: Mapping[str, Any], text: str
) -> tuple[list[int], list[int]]:
    representation = protocol["representation"]
    return _encode(tokenizer, text), _character_features(
        text,
        int(representation["character_hash_buckets"]),
        int(representation["character_ngram_minimum"]),
        int(representation["character_ngram_maximum"]),
        int(representation["hash_seed"]),
    )


def _data(root: Path, protocol: Mapping[str, Any], tokenizer: Any) -> list[dict[str, Any]]:
    rows = []
    for row in load_phase1_ir((root / protocol["phase1_ir"]).resolve()):
        prompt = str(row["normalized_acquisition_prompt"])
        lines = prompt.splitlines()
        if len(lines) < 2 or not lines[0].strip():
            raise Phase3Error("acquisition record lacks a metadata/body boundary")
        metadata = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        body_bpe, body_character = _features(tokenizer, protocol, body)
        metadata_bpe, metadata_character = _features(tokenizer, protocol, metadata)
        rows.append(
            {
                "record_id": str(row["ir_record_id"]),
                "capability": str(row["capability"]),
                "body_bpe": body_bpe,
                "body_character": body_character,
                "metadata_bpe": metadata_bpe,
                "metadata_character": metadata_character,
                "metadata": metadata,
            }
        )
    return rows


def _bag(
    sequences: Sequence[Sequence[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = []
    flattened = []
    for sequence in sequences:
        if not sequence:
            raise Phase3Error("empty sparse-router feature sequence")
        offsets.append(len(flattened))
        flattened.extend(sequence)
    return (
        torch.tensor(flattened, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


def _model(protocol: Mapping[str, Any], vocabulary: int) -> SparseRouter:
    return SparseRouter(
        vocabulary,
        int(protocol["representation"]["character_hash_buckets"]),
        len(CAPABILITIES) + 1,
    )


def inventory(root: Path, path: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    rows = _data(root, protocol, tokenizer)
    model = _model(protocol, tokenizer.vocab_size)
    parameters = sum(value.numel() for value in model.parameters())
    if parameters != int(protocol["training"]["trainable_parameters"]):
        raise Phase3Error(f"sparse-router parameter count changed: {parameters}")
    return {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "records": len(rows),
        "capabilities": len(CAPABILITIES),
        "trainable_parameters": parameters,
        "maximum_body_bpe_actions": max(len(row["body_bpe"]) for row in rows),
        "maximum_body_character_features": max(
            len(row["body_character"]) for row in rows
        ),
        "unique_metadata_segments": len({row["metadata"] for row in rows}),
        "teacher_outputs_added": 0,
        "final_test_accessed": False,
    }


def train(root: Path, path: Path, output: Path) -> Mapping[str, Any]:
    protocol, protocol_hash = load_protocol(root, path)
    if output.exists() or not torch.cuda.is_available():
        raise Phase3Error("sparse-router output exists or CUDA unavailable")
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
        raise Phase3Error("sparse-router parameter count changed")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=float(config["weight_decay"]),
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
        bpe_sequences = []
        character_sequences = []
        targets = []
        for row in batch:
            bpe_sequences.extend((row["body_bpe"], row["metadata_bpe"]))
            character_sequences.extend(
                (row["body_character"], row["metadata_character"])
            )
            targets.extend((label_to_id[row["capability"]], label_to_id[METADATA]))
            sequence_hash.update(row["record_id"].encode() + b"\n")
        bpe_ids, bpe_offsets = _bag(bpe_sequences, device)
        character_ids, character_offsets = _bag(character_sequences, device)
        target = torch.tensor(targets, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(bpe_ids, bpe_offsets, character_ids, character_offsets)
        loss = F.cross_entropy(logits, target, weight=class_weights)
        loss.backward()
        optimizer.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % int(config["curve_interval"]) == 0:
            predictions = logits.argmax(dim=-1)
            body_positions = torch.arange(len(targets), device=device) % 2 == 0
            value = {
                "step": step,
                "loss": float(loss.detach()),
                "body_accuracy": float(
                    predictions[body_positions]
                    .eq(target[body_positions])
                    .float()
                    .mean()
                ),
                "metadata_accuracy": float(
                    predictions[~body_positions]
                    .eq(target[~body_positions])
                    .float()
                    .mean()
                ),
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
            {"vocabulary": tokenizer.vocab_size, **protocol["representation"]},
            sort_keys=True,
            indent=2,
        ).encode()
        + b"\n",
    )
    metadata: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-sparse-router-candidate/1",
        "status": "TRAINED_SPARSE_ROUTER_GATE_ONLY",
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
) -> tuple[SparseRouter, Any]:
    _, _, tokenizer_type, _, _ = _layercake_api(root, protocol)
    tokenizer = _tokenizer(root, protocol, tokenizer_type)
    model = _model(protocol, tokenizer.vocab_size)
    model.load_state_dict(
        load_file(str(candidate / "router.safetensors"), device="cuda"), strict=True
    )
    return model.cuda().eval(), tokenizer


@torch.inference_mode()
def _score(
    model: SparseRouter,
    tokenizer: Any,
    protocol: Mapping[str, Any],
    texts: Sequence[str],
) -> torch.Tensor:
    values = [_features(tokenizer, protocol, text) for text in texts]
    bpe_ids, bpe_offsets = _bag([value[0] for value in values], torch.device("cuda"))
    character_ids, character_offsets = _bag(
        [value[1] for value in values], torch.device("cuda")
    )
    return model(bpe_ids, bpe_offsets, character_ids, character_offsets)


@torch.inference_mode()
def _route(
    model: SparseRouter, tokenizer: Any, protocol: Mapping[str, Any], text: str
) -> tuple[str, list[dict[str, Any]]]:
    segments = _semantic_segments(text)
    probabilities = _score(model, tokenizer, protocol, segments).softmax(dim=-1).cpu()
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
        raise Phase3Error("sparse-router evaluation identity failed")
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
            prediction, details = _route(model, tokenizer, protocol, text)
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
        metadata_prediction = int(
            _score(model, tokenizer, protocol, [metadata_segment]).argmax(dim=-1)[0]
        )
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
        "format": "abi-capability-compiler-phase3-sparse-router-decision/1",
        "status": (
            "PASS_SPARSE_ROUTER_GATE_REPLICATION_OPEN"
            if passed
            else "FAIL_SPARSE_ROUTER_GATE_ARCHITECTURE_CLOSED"
        ),
        "protocol": {
            "path": "ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_ROUTER_PROTOCOL_V45.json",
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
        default="ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_ROUTER_PROTOCOL_V45.json",
    )
    parser.add_argument(
        "--candidate-dir",
        default="results/abi_capability_compiler_phase3_sparse_router/development_v45/R0-seed240045",
    )
    parser.add_argument(
        "--output-dir",
        default="results/abi_capability_compiler_phase3_sparse_router/evaluation_v45/R0-seed240045",
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
