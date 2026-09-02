"""Shared custody and row helpers for the R13-B finite-table control."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from experiments.foreign_teacher_r12.public_preflight import _atomic_rows
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    generate_rows,
)
from experiments.native_transfer_r8.native_host import FrozenNeuralHost


class R13Error(RuntimeError):
    """Raised when R13 custody, evidence, or a registered gate fails."""


def json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R13Error(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise R13Error(f"expected JSON object: {path}")
    return value


def write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise R13Error(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(dict(value), indent=2, sort_keys=True).encode() + b"\n")


def write_jsonl_once(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise R13Error(f"immutable output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_json_bytes(dict(row)) for row in rows))


def evidence_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def capability_rows(
    config: Mapping[str, Any], capabilities: Sequence[Any]
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    data = config["data"]
    training: list[list[dict[str, Any]]] = []
    evaluation: list[list[dict[str, Any]]] = []
    atomic: list[list[dict[str, Any]]] = []
    for index, capability in enumerate(capabilities):
        train = generate_rows(
            capability,
            split="source_train",
            rows=int(data["training_rows_per_capability"]),
            depths=data["training_depths"],
            seed=int(data["training_seed"]) + 1009 * index,
        )
        test = generate_rows(
            capability,
            split="r13_heldout_evaluation",
            rows=int(data["evaluation_rows_per_capability"]),
            depths=data["evaluation_depths"],
            seed=int(data["evaluation_seed"]) + 9001 * index,
        )
        train_hashes = {str(row["prompt_sha256"]) for row in train}
        test_hashes = {str(row["prompt_sha256"]) for row in test}
        if len(train_hashes) != len(train) or len(test_hashes) != len(test):
            raise R13Error("duplicate source or evaluation prompt")
        if train_hashes & test_hashes:
            raise R13Error("source-training and evaluation prompts overlap")
        atoms = []
        for atom_index, row in enumerate(_atomic_rows(capability)):
            prompt = str(row["prompt"])
            atoms.append(
                {
                    **row,
                    "capability_id": capability.capability_id,
                    "row_id": hashlib.sha256(
                        canonical_json_bytes(
                            {
                                "capability_id": capability.capability_id,
                                "split": "r13_atomic",
                                "index": atom_index,
                            }
                        )
                    ).hexdigest(),
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "split": "r13_atomic",
                    "depth": 1,
                }
            )
        training.append(train)
        evaluation.append(test)
        atomic.append(atoms)
    return training, evaluation, atomic


@torch.inference_mode()
def observe_source(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    capability_id: str,
    condition: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    correct = 0
    canonical_correct = 0
    prediction_ids: list[int] = []
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        native = logits.argmax(dim=-1)
        canonical_logits = logits.index_select(
            -1, torch.tensor(host.target_token_ids, device=logits.device)
        )
        canonical = canonical_logits.argmax(dim=-1)
        probabilities = torch.softmax(canonical_logits, dim=-1)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        answers = torch.tensor([int(row["answer"]) for row in batch], device=logits.device)
        correct += int((native == targets).sum())
        canonical_correct += int((canonical == answers).sum())
        prediction_ids.extend(int(value) for value in native.cpu().tolist())
        for row, native_id, canonical_id, probability in zip(
            batch, native, canonical, probabilities
        ):
            observations.append(
                {
                    "capability_id": capability_id,
                    "condition": condition,
                    "row_id": str(row["row_id"]),
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "answer": int(row["answer"]),
                    "native_prediction_token_id": int(native_id.cpu()),
                    "canonical_prediction": int(canonical_id.cpu()),
                    "canonical_probabilities": [
                        float(value) for value in probability.float().cpu()
                    ],
                }
            )
    metrics = {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "canonical_correct": canonical_correct,
        "canonical_accuracy": canonical_correct / len(rows),
        "prediction_ids_sha256": hashlib.sha256(
            ",".join(str(value) for value in prediction_ids).encode()
        ).hexdigest(),
    }
    return metrics, observations
