"""Preregistered source acquisition for the R13-B finite-table control."""

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

from .core import R13Error


def train_until_atomic_stable(
    host: FrozenNeuralHost,
    adapters: GenericRecipientAdapterSet,
    training_rows: Sequence[Mapping[str, Any]],
    atomic_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_steps: int,
    evaluation_interval: int,
    stable_evaluations: int,
    learning_rate: float,
    batch_size: int,
    evaluation_batch_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Train without touching the depth-6/7 held-out evaluation rows."""
    if maximum_steps <= 0 or evaluation_interval <= 0 or stable_evaluations <= 0:
        raise R13Error("invalid source acquisition schedule")
    permitted = {id(parameter) for parameter in adapters.parameters()}
    for parameter in host.model.parameters():
        parameter.requires_grad_(id(parameter) in permitted)
    parameters = adapters.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = random.Random(seed)
    started = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    consecutive = 0
    first_loss = None
    final_loss = None
    selected_state = None
    selected_step = None
    for step in range(1, maximum_steps + 1):
        batch = sample_training_batch(
            training_rows,
            batch_size=batch_size,
            generator=generator,
            strategy="depth_balanced",
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise R13Error("source acquisition loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        final_loss = value
        if step % evaluation_interval == 0 or step == maximum_steps:
            atomic = evaluate(host, atomic_rows, batch_size=evaluation_batch_size)
            consecutive = consecutive + 1 if atomic["accuracy"] == 1.0 else 0
            observation = {
                "step": step,
                "atomic": atomic,
                "consecutive_exact_atomic_evaluations": consecutive,
            }
            checkpoints.append(observation)
            print(observation, flush=True)
            if consecutive >= stable_evaluations:
                selected_state = adapters.state()
                selected_step = step
                break
    if selected_state is None:
        selected_state = adapters.state()
    adapters.verify_base_frozen()
    return selected_state, {
        "maximum_steps": maximum_steps,
        "optimizer_steps": step,
        "selected_step": selected_step,
        "selection_used_heldout_rows": False,
        "stable_atomic_evaluations_required": stable_evaluations,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "sampling_strategy": "depth_balanced",
        "first_loss": first_loss,
        "final_loss": final_loss,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "wall_seconds": time.perf_counter() - started,
        "checkpoints": checkpoints,
    }
