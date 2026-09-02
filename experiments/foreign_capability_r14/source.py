"""Conventional Qwen acquisition and bounded observation for R14."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from experiments.foreign_teacher_r12.teacher import evaluate, sample_training_batch
from experiments.native_transfer_r8.native_host import (
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)


class R14SourceError(RuntimeError):
    """Raised when source acquisition or observation violates the protocol."""


def train_fixed_schedule(
    host: FrozenNeuralHost,
    adapters: GenericRecipientAdapterSet,
    training_rows: Sequence[Mapping[str, Any]],
    public_development_rows: Sequence[Mapping[str, Any]] | None,
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    evaluation_interval: int,
    evaluation_batch_size: int,
    seed: int,
    objective: str = "full_vocabulary",
) -> dict[str, Any]:
    """Run a fixed schedule; held-out runs provide no development rows."""
    if steps <= 0 or evaluation_interval <= 0:
        raise R14SourceError("invalid fixed source schedule")
    permitted = {id(parameter) for parameter in adapters.parameters()}
    for parameter in host.model.parameters():
        parameter.requires_grad_(id(parameter) in permitted)
    parameters = adapters.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = random.Random(int(seed))
    started = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    first_loss = None
    final_loss = None
    for step in range(1, steps + 1):
        batch = sample_training_batch(
            training_rows,
            batch_size=batch_size,
            generator=generator,
            strategy="depth_balanced",
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        answers = torch.tensor([int(row["answer"]) for row in batch], device=host.device)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        if objective == "full_vocabulary":
            loss = F.cross_entropy(logits, targets)
        elif objective == "canonical_plus_native":
            canonical_ids = torch.tensor(host.target_token_ids, device=host.device)
            canonical_logits = logits.index_select(-1, canonical_ids)
            loss = F.cross_entropy(canonical_logits, answers) + 0.1 * F.cross_entropy(
                logits, targets
            )
        else:
            raise R14SourceError(f"unknown source objective: {objective}")
        if not torch.isfinite(loss):
            raise R14SourceError("source loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        first_loss = final_loss if first_loss is None else first_loss
        if step % evaluation_interval == 0 or step == steps:
            checkpoint: dict[str, Any] = {"step": step, "loss": final_loss}
            if public_development_rows is not None:
                checkpoint["public_development"] = evaluate(
                    host,
                    public_development_rows,
                    batch_size=evaluation_batch_size,
                )
            checkpoints.append(checkpoint)
            print(checkpoint, flush=True)
    adapters.verify_base_frozen()
    adapters.freeze()
    return {
        "schedule_selection_used_heldout": False,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "sampling": "depth_balanced",
        "objective": objective,
        "optimizer": "AdamW",
        "first_loss": first_loss,
        "final_loss": final_loss,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "wall_seconds": time.perf_counter() - started,
        "checkpoints": checkpoints,
    }


@torch.inference_mode()
def observe_queries(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    native_correct = 0
    canonical_correct = 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        native = logits.argmax(dim=-1)
        canonical_probabilities = host.canonical_probabilities(logits)
        canonical = canonical_probabilities.argmax(dim=-1)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        answers = torch.tensor([int(row["answer"]) for row in batch], device=host.device)
        native_correct += int(native.eq(targets).sum())
        canonical_correct += int(canonical.eq(answers).sum())
        for row, probabilities in zip(batch, canonical_probabilities):
            observations.append(
                {
                    "row_id": str(row["row_id"]),
                    "prompt_sha256": str(row["prompt_sha256"]),
                    "start": int(row["start"]),
                    "program": [int(value) for value in row["program"]],
                    "canonical_probabilities": [
                        float(value) for value in probabilities.float().cpu()
                    ],
                }
            )
    return observations, {
        "rows": len(rows),
        "native_correct": native_correct,
        "native_accuracy": native_correct / len(rows),
        "canonical_correct": canonical_correct,
        "canonical_accuracy": canonical_correct / len(rows),
    }
