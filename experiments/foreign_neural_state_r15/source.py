"""Conventional Qwen learning events with auditable before/after weight deltas."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from experiments.foreign_capability_r14.source import train_fixed_schedule
from experiments.native_transfer_r8.native_host import (
    FrozenNeuralHost,
    GenericRecipientAdapterSet,
    tensor_state_sha256,
)


class R15SourceError(RuntimeError):
    """Raised when an R15 source learning event violates its boundary."""


class TargetedOutputLoRA(nn.Module):
    """LoRA update to Qwen's canonical output rows.

    The base output projection is unchanged.  The effective before/after model
    weight delta is ``B @ A.T / rank`` for the eight registered output rows.
    Restricting the adapter to scored rows makes repeated public learning
    events tractable while retaining ordinary gradient acquisition in Qwen.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        target_ids: Sequence[int],
        rank: int,
        initialization_seed: int,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise R15SourceError("invalid Qwen output adapter")
        ids = tuple(int(value) for value in target_ids)
        if len(ids) != 8 or len(set(ids)) != 8:
            raise R15SourceError("canonical target-token inventory changed")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.register_buffer("target_ids", torch.tensor(ids, dtype=torch.long))
        self.a = nn.Parameter(torch.empty(base.in_features, self.rank, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(self.rank, len(ids), dtype=torch.float32))
        devices = [base.weight.device.index] if base.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(initialization_seed))
            nn.init.normal_(self.a, mean=0.0, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.base(inputs)
        update = torch.matmul(torch.matmul(inputs.float(), self.a), self.b) / self.rank
        delta = torch.zeros_like(logits)
        delta.index_copy_(-1, self.target_ids.to(logits.device), update.to(logits.dtype))
        return logits + delta

    def effective_delta(self) -> torch.Tensor:
        """Return the actual eight-row output-weight change, never LoRA factors."""
        return torch.matmul(self.b.detach().t(), self.a.detach().t()).float() / self.rank


class QwenLearningEvent:
    """Install and reset one conventional source-side output LoRA."""

    def __init__(
        self,
        host: FrozenNeuralHost,
        *,
        rank: int,
        initialization_seed: int,
    ) -> None:
        output = host.model.get_output_embeddings()
        if not isinstance(output, nn.Linear):
            raise R15SourceError("Qwen output embedding is not linear")
        self.host = host
        self.base_model_sha256 = host.model_state_sha256
        self.adapter = TargetedOutputLoRA(
            output,
            target_ids=host.target_token_ids,
            rank=rank,
            initialization_seed=initialization_seed,
        ).to(host.device)
        host.model.set_output_embeddings(self.adapter)
        self.before_state = {
            "a": self.adapter.a.detach().cpu().clone(),
            "b": self.adapter.b.detach().cpu().clone(),
        }
        self.before_delta_sha256 = self.delta_sha256()

    def reset(self) -> None:
        with torch.no_grad():
            self.adapter.a.copy_(self.before_state["a"].to(self.adapter.a))
            self.adapter.b.copy_(self.before_state["b"].to(self.adapter.b))
        self.adapter.train()

    def parameters(self) -> list[nn.Parameter]:
        return [self.adapter.a, self.adapter.b]

    def state(self) -> dict[str, torch.Tensor]:
        return {
            "a": self.adapter.a.detach().cpu().contiguous(),
            "b": self.adapter.b.detach().cpu().contiguous(),
        }

    def load_state(self, state: Mapping[str, torch.Tensor]) -> None:
        if (
            set(state) != {"a", "b"}
            or tuple(state["a"].shape) != tuple(self.adapter.a.shape)
            or tuple(state["b"].shape) != tuple(self.adapter.b.shape)
            or not all(torch.isfinite(value).all() for value in state.values())
        ):
            raise R15SourceError("source adapter state contract changed")
        with torch.no_grad():
            self.adapter.a.copy_(state["a"].to(self.adapter.a))
            self.adapter.b.copy_(state["b"].to(self.adapter.b))
        self.adapter.eval()

    def state_sha256(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state().items()):
            digest.update(name.encode() + b"\0")
            digest.update(str(value.dtype).encode("ascii") + b"\0")
            digest.update(str(list(value.shape)).encode("ascii") + b"\0")
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def delta(self) -> torch.Tensor:
        return self.adapter.effective_delta().cpu().contiguous()

    def delta_sha256(self) -> str:
        value = self.delta()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(str(list(value.shape)).encode("ascii") + b"\0")
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def verify_base_frozen(self) -> None:
        state = {}
        for name, value in self.host.model.state_dict().items():
            if name in {"lm_head.a", "lm_head.b", "lm_head.target_ids"}:
                continue
            if name == "lm_head.base.weight":
                state["lm_head.weight"] = value
            else:
                state[name] = value
        if tensor_state_sha256(state) != self.base_model_sha256:
            raise R15SourceError("Qwen base weights changed during learning event")


class FullQwenLoRALearningEvent:
    """A resettable all-linear Qwen LoRA event with fixed weight-delta probes."""

    def __init__(
        self,
        host: FrozenNeuralHost,
        *,
        rank: int,
        initialization_seed: int,
        projections_per_module: int,
    ) -> None:
        if projections_per_module <= 0:
            raise R15SourceError("weight-delta projection count must be positive")
        torch.manual_seed(int(initialization_seed))
        self.host = host
        self.adapters = GenericRecipientAdapterSet(host, rank=rank)
        self.base_model_sha256 = host.model_state_sha256
        self.before_state = {
            name: value.detach().cpu().clone() for name, value in self.adapters.state().items()
        }
        self.before_state_sha256 = self._state_sha256(self.before_state)
        self.projections_per_module = int(projections_per_module)
        self._probes: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        for name, module in sorted(self.adapters.modules.items()):
            pairs = []
            for index in range(self.projections_per_module):
                seed_material = hashlib.sha256(
                    f"abi-r15a-weight-probe\0{name}\0{index}".encode()
                ).digest()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int.from_bytes(seed_material[:8], "big") % (2**63 - 1))
                left = torch.randint(
                    0,
                    2,
                    (module.a.shape[0],),
                    generator=generator,
                    dtype=torch.float32,
                )
                right = torch.randint(
                    0,
                    2,
                    (module.b.shape[1],),
                    generator=generator,
                    dtype=torch.float32,
                )
                left = (2 * left - 1) / max(1.0, float(left.numel()) ** 0.5)
                right = (2 * right - 1) / max(1.0, float(right.numel()) ** 0.5)
                pairs.append((left.to(module.a.device), right.to(module.b.device)))
            self._probes[name] = pairs

    @staticmethod
    def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
        digest = hashlib.sha256()
        for name in sorted(state):
            value = state[name].detach().cpu().contiguous()
            digest.update(name.encode() + b"\0")
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def reset(self) -> None:
        self.adapters.load_state(self.before_state)

    def delta_sha256(self) -> str:
        return self._state_sha256(self.adapters.state())

    def delta(self) -> torch.Tensor:
        """Fixed bilinear sketch of actual effective LoRA weight changes."""
        features = []
        for name, module in sorted(self.adapters.modules.items()):
            a = module.a.detach().float()
            b = module.b.detach().float()
            before_a = self.before_state[f"{name}/lora_a"].to(a)
            before_b = self.before_state[f"{name}/lora_b"].to(b)
            for left, right in self._probes[name]:
                after_value = torch.dot(torch.matmul(left, a), torch.matmul(b, right))
                before_value = torch.dot(
                    torch.matmul(left, before_a), torch.matmul(before_b, right)
                )
                features.append((after_value - before_value) / module.rank)
            after_gram = torch.matmul(a.t(), a) @ torch.matmul(b, b.t())
            before_gram = torch.matmul(before_a.t(), before_a) @ torch.matmul(
                before_b, before_b.t()
            )
            features.append(
                (after_gram.trace().clamp_min(0).sqrt() - before_gram.trace().clamp_min(0).sqrt())
                / module.rank
            )
        return torch.stack(features).detach().cpu().contiguous()

    def verify_base_frozen(self) -> None:
        self.adapters.verify_base_frozen()


def train_full_lora_learning_event(
    event: FullQwenLoRALearningEvent,
    training_rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    event.reset()
    result = train_fixed_schedule(
        event.host,
        event.adapters,
        training_rows,
        None,
        steps=steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        evaluation_interval=steps,
        evaluation_batch_size=batch_size,
        seed=seed,
        objective="canonical_plus_native",
    )
    event.verify_base_frozen()
    return {
        **result,
        "before_state_sha256": event.before_state_sha256,
        "after_state_sha256": event.delta_sha256(),
        "delta_features": int(event.delta().numel()),
        "projections_per_module": event.projections_per_module,
    }


def train_learning_event(
    host: FrozenNeuralHost,
    event: QwenLearningEvent,
    training_rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Train one fixed-schedule event from the identical before state."""
    if steps <= 0 or batch_size <= 0 or not training_rows:
        raise R15SourceError("invalid source learning schedule")
    event.reset()
    for parameter in host.model.parameters():
        parameter.requires_grad_(False)
    for parameter in event.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(event.parameters(), lr=learning_rate, weight_decay=0.0)
    generator = random.Random(int(seed))
    started = time.perf_counter()
    first_loss: float | None = None
    final_loss: float | None = None
    for _step in range(steps):
        indices = [generator.randrange(len(training_rows)) for _ in range(batch_size)]
        batch = [training_rows[index] for index in indices]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        answers = torch.tensor([int(row["answer"]) for row in batch], device=host.device)
        canonical_ids = torch.tensor(host.target_token_ids, device=host.device)
        canonical_logits = logits.index_select(-1, canonical_ids)
        loss = F.cross_entropy(canonical_logits, answers)
        if not torch.isfinite(loss):
            raise R15SourceError("source loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(event.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if first_loss is None:
            first_loss = final_loss
    event.verify_base_frozen()
    for parameter in event.parameters():
        parameter.requires_grad_(False)
    event.adapter.eval()
    return {
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "wall_seconds": time.perf_counter() - started,
        "before_delta_sha256": event.before_delta_sha256,
        "after_delta_sha256": event.delta_sha256(),
        "delta_elements": int(event.delta().numel()),
        "trainable_parameters": sum(value.numel() for value in event.parameters()),
    }


@torch.inference_mode()
def cache_output_training_inputs(
    host: FrozenNeuralHost,
    event: QwenLearningEvent,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Cache frozen Qwen residuals and base logits for source training only."""
    prompts = [str(row["prompt"]) for row in rows]
    if len(prompts) != len(set(prompts)):
        raise R15SourceError("source cache prompts are not unique")
    encoded = host.encode(prompts)
    output = host.model.model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=False,
        return_dict=True,
    )
    last = encoded["attention_mask"].sum(dim=1) - 1
    batch = torch.arange(len(prompts), device=host.device)
    hidden = output.last_hidden_state[batch, last].float()
    base_logits = event.adapter.base(hidden.to(event.adapter.base.weight.dtype)).float()
    canonical_ids = event.adapter.target_ids.to(base_logits.device)
    canonical = base_logits.index_select(-1, canonical_ids)
    return {
        prompt: (hidden[index].detach(), canonical[index].detach())
        for index, prompt in enumerate(prompts)
    }


def train_cached_output_learning_event(
    event: QwenLearningEvent,
    training_rows: Sequence[Mapping[str, Any]],
    cache: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Train the same output LoRA using exactly cached frozen-Qwen residuals."""
    if steps <= 0 or not training_rows:
        raise R15SourceError("invalid cached source learning schedule")
    ordered = sorted(training_rows, key=lambda row: str(row["prompt"]))
    if any(str(row["prompt"]) not in cache for row in ordered):
        raise R15SourceError("source cache does not cover the learning event")
    hidden = torch.stack([cache[str(row["prompt"])][0] for row in ordered])
    base_logits = torch.stack([cache[str(row["prompt"])][1] for row in ordered])
    answers = torch.tensor(
        [int(row["answer"]) for row in ordered],
        dtype=torch.long,
        device=hidden.device,
    )
    event.reset()
    for parameter in event.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(event.parameters(), lr=learning_rate, weight_decay=0.0)
    started = time.perf_counter()
    first_loss: float | None = None
    final_loss: float | None = None
    for _step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        update = torch.matmul(torch.matmul(hidden, event.adapter.a), event.adapter.b)
        logits = base_logits + update / event.adapter.rank
        loss = F.cross_entropy(logits, answers)
        if not torch.isfinite(loss):
            raise R15SourceError("cached source loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(event.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if first_loss is None:
            first_loss = final_loss
    event.verify_base_frozen()
    for parameter in event.parameters():
        parameter.requires_grad_(False)
    event.adapter.eval()
    with torch.inference_mode():
        update = torch.matmul(torch.matmul(hidden, event.adapter.a), event.adapter.b)
        predictions = (base_logits + update / event.adapter.rank).argmax(dim=-1)
        correct = int(predictions.eq(answers).sum())
    return {
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "batch_size": len(ordered),
        "sampling": "full_balanced_atomic_batch",
        "frozen_qwen_residual_cache": True,
        "cache_available_to_extractor": False,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "source_atomic_correct": correct,
        "source_atomic_rows": len(ordered),
        "source_atomic_accuracy": correct / len(ordered),
        "wall_seconds": time.perf_counter() - started,
        "before_delta_sha256": event.before_delta_sha256,
        "after_delta_sha256": event.delta_sha256(),
        "delta_elements": int(event.delta().numel()),
        "trainable_parameters": sum(value.numel() for value in event.parameters()),
    }


@torch.inference_mode()
def canonical_accuracy(
    host: FrozenNeuralHost,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    correct = 0
    for offset in range(0, len(rows), batch_size):
        batch = rows[offset : offset + batch_size]
        logits, _ = host.logits([str(row["prompt"]) for row in batch], prefix=None)
        predictions = host.canonical_probabilities(logits).argmax(dim=-1)
        answers = torch.tensor([int(row["answer"]) for row in batch], device=host.device)
        correct += int(predictions.eq(answers).sum())
    return {"rows": len(rows), "correct": correct, "accuracy": correct / len(rows)}
