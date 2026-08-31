"""Conventional Qwen source-local fine-tuning for the R12-A frontend gate."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from experiments.native_transfer_r8.native_host import (
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
)


class R12TeacherError(RuntimeError):
    """Raised when the conventional-teacher custody or training gate changes."""


def sample_training_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    generator: random.Random,
    strategy: str,
) -> list[Mapping[str, Any]]:
    if strategy == "row_uniform":
        return [rows[generator.randrange(len(rows))] for _ in range(batch_size)]
    if strategy != "depth_balanced":
        raise R12TeacherError(f"unknown teacher sampling strategy: {strategy}")
    by_depth: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_depth.setdefault(int(row["depth"]), []).append(row)
    depths = sorted(by_depth)
    if not depths or batch_size % len(depths):
        raise R12TeacherError("depth-balanced batch must divide evenly across depths")
    per_depth = batch_size // len(depths)
    batch = [
        by_depth[depth][generator.randrange(len(by_depth[depth]))]
        for depth in depths
        for _ in range(per_depth)
    ]
    generator.shuffle(batch)
    return batch


@torch.inference_mode()
def evaluate(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    correct = 0
    canonical_correct = 0
    predictions = []
    nll = 0.0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        predicted = logits.argmax(dim=-1)
        canonical = logits.index_select(
            -1, torch.tensor(host.target_token_ids, device=logits.device)
        ).argmax(dim=-1)
        answers = torch.tensor([int(row["answer"]) for row in batch], device=logits.device)
        correct += int((predicted == targets).sum())
        canonical_correct += int((canonical == answers).sum())
        nll += float(F.cross_entropy(logits, targets, reduction="sum"))
        predictions.extend(int(value) for value in predicted.cpu().tolist())
    return {
        "rows": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "canonical_accuracy": canonical_correct / len(rows),
        "mean_nll": nll / len(rows),
        "prediction_ids_sha256": hashlib.sha256(
            (",".join(str(value) for value in predictions)).encode()
        ).hexdigest(),
    }


def train(
    host: FrozenNeuralHost,
    adapters: GenericRecipientAdapterSet,
    training_rows: Sequence[Mapping[str, Any]],
    evaluation_rows: Sequence[Mapping[str, Any]],
    atomic_rows: Sequence[Mapping[str, Any]],
    *,
    maximum_steps: int,
    evaluation_interval: int,
    learning_rate: float,
    batch_size: int,
    evaluation_batch_size: int,
    seed: int,
    sampling_strategy: str = "row_uniform",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    if maximum_steps <= 0 or evaluation_interval <= 0:
        raise R12TeacherError("invalid teacher training schedule")
    permitted = {id(parameter) for parameter in adapters.parameters()}
    for parameter in host.model.parameters():
        parameter.requires_grad_(id(parameter) in permitted)
    parameters = adapters.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
    generator = random.Random(seed)
    started = time.perf_counter()
    first_loss = None
    final_loss = None
    checkpoints = []
    selected_state = None
    selected_step = None
    for step in range(1, maximum_steps + 1):
        batch = sample_training_batch(
            training_rows,
            batch_size=batch_size,
            generator=generator,
            strategy=sampling_strategy,
        )
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        targets = host.target_ids([int(row["answer"]) for row in batch])
        loss = F.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise R12TeacherError("conventional teacher loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        value = float(loss.detach())
        first_loss = value if first_loss is None else first_loss
        final_loss = value
        if step % evaluation_interval == 0 or step == maximum_steps:
            evaluation = evaluate(host, evaluation_rows, batch_size=evaluation_batch_size)
            atomic = evaluate(host, atomic_rows, batch_size=evaluation_batch_size)
            observation = {"step": step, "evaluation": evaluation, "atomic": atomic}
            checkpoints.append(observation)
            print(observation, flush=True)
            if evaluation["accuracy"] == 1.0 and atomic["accuracy"] == 1.0:
                selected_state = adapters.state()
                selected_step = step
                break
    adapters.verify_base_frozen()
    if selected_state is None:
        selected_state = adapters.state()
    return selected_state, {
        "maximum_steps": maximum_steps,
        "optimizer_steps": step,
        "selected_step": selected_step,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "optimizer": "AdamW",
        "sampling_strategy": sampling_strategy,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "trainable_parameters": sum(value.numel() for value in parameters),
        "wall_seconds": time.perf_counter() - started,
        "checkpoints": checkpoints,
    }
