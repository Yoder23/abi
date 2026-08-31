"""Capability package, neural executor, training, and native host codec for R11."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from experiments.copy_paste_r10.run import R10FrozenNeuralHost
from experiments.copy_paste_r10.runtime import CanonicalTransitionVM
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes
from experiments.native_transfer_r8.native_host import SPECS, module_sha256

PACKAGE_FORMAT = "abi-native-neural-isa-r11-transition/1"
PACKAGE_KEYS = {
    "format",
    "family",
    "neural_isa",
    "transition_dtype",
    "transition_hex",
    "transition_sha256",
    "transition_shape",
    "provenance",
}


class R11Error(RuntimeError):
    """Raised when an R11 neural-ISA invariant changes."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transition_bytes(transition: torch.Tensor) -> bytes:
    value = transition.detach().cpu().float().contiguous()
    if tuple(value.shape) != (3, 8, 8) or not torch.isfinite(value).all():
        raise R11Error("transition shape or numerics changed")
    if (value < 0).any() or not torch.allclose(
        value.sum(dim=-1), torch.ones(3, 8), atol=1e-5, rtol=0
    ):
        raise R11Error("transition is not a normalized neural state")
    return struct.pack("<192f", *[float(item) for item in value.flatten().tolist()])


def build_package(transition: torch.Tensor, provenance: Mapping[str, str]) -> dict[str, Any]:
    if set(provenance) != {"teacher_before_sha256", "teacher_after_sha256"}:
        raise R11Error("package provenance schema changed")
    raw = transition_bytes(transition)
    return {
        "format": PACKAGE_FORMAT,
        "family": "opaque_modular_micro_language/1",
        "neural_isa": "recurrent-transition-neural-isa/1",
        "transition_dtype": "float32-little-endian",
        "transition_hex": raw.hex(),
        "transition_sha256": sha256_bytes(raw),
        "transition_shape": [3, 8, 8],
        "provenance": dict(provenance),
    }


def write_package_once(
    directory: Path, transition: torch.Tensor, provenance: Mapping[str, str]
) -> dict[str, Any]:
    package = build_package(transition, provenance)
    content = canonical_json_bytes(package)
    digest = sha256_bytes(content)
    path = directory / f"sha256-{digest}.abipkg"
    if path.exists():
        raise R11Error(f"immutable package exists: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.name,
        "sha256": digest,
        "bytes": len(content),
        "transition_sha256": package["transition_sha256"],
    }


def load_package(path: Path) -> tuple[dict[str, Any], torch.Tensor]:
    try:
        content = path.read_bytes()
        value = json.loads(content)
    except (OSError, json.JSONDecodeError) as exc:
        raise R11Error(f"package unavailable: {path}") from exc
    if not isinstance(value, dict) or set(value) != PACKAGE_KEYS:
        raise R11Error("package schema changed")
    if (
        content != canonical_json_bytes(value)
        or sha256_bytes(content) != path.stem.removeprefix("sha256-")
        or value.get("format") != PACKAGE_FORMAT
        or value.get("family") != "opaque_modular_micro_language/1"
        or value.get("neural_isa") != "recurrent-transition-neural-isa/1"
        or value.get("transition_dtype") != "float32-little-endian"
        or value.get("transition_shape") != [3, 8, 8]
    ):
        raise R11Error("package identity or ABI changed")
    forbidden = {
        "prompt",
        "answer",
        "row_id",
        "solver",
        "model_id",
        "tokenizer_id",
        "hidden_width",
        "host_matrix",
        "capability_id",
        "seed",
        "offsets",
    }
    if forbidden & set(value):
        raise R11Error("package contains forbidden host or evaluation material")
    try:
        raw = bytes.fromhex(str(value["transition_hex"]))
    except ValueError as exc:
        raise R11Error("package transition is not hexadecimal") from exc
    if len(raw) != 768 or sha256_bytes(raw) != value.get("transition_sha256"):
        raise R11Error("package transition identity changed")
    transition = torch.tensor(struct.unpack("<192f", raw), dtype=torch.float32).reshape(3, 8, 8)
    transition_bytes(transition)
    return value, transition


class RecurrentTransitionNeuralISA(nn.Module):
    """Capability-blind differentiable executor frozen before held-out reveal."""

    learned_parameters = 0
    abi = "recurrent-transition-neural-isa/1"

    def forward(self, transition: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        if tuple(transition.shape) != (3, 8, 8):
            raise R11Error("neural ISA transition shape changed")
        outputs = []
        for prompt in prompts:
            start, program = CanonicalTransitionVM.parse(str(prompt))
            state = torch.nn.functional.one_hot(
                torch.tensor(start, device=transition.device), num_classes=8
            ).to(transition)
            for operation in program:
                state = torch.matmul(state, transition[operation])
            outputs.append(state / state.sum().clamp_min(1e-12))
        return torch.stack(outputs)


def _execute_tensor_programs(
    transition: torch.Tensor, starts: torch.Tensor, programs: torch.Tensor
) -> torch.Tensor:
    state = torch.nn.functional.one_hot(starts, num_classes=8).to(transition)
    for position in range(programs.shape[1]):
        selected = transition.index_select(0, programs[:, position])
        state = torch.bmm(state.unsqueeze(1), selected).squeeze(1)
    return state


def train_teacher_transition(
    rows: Sequence[Mapping[str, Any]],
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    selected = torch.device(device)
    logits = nn.Parameter(torch.zeros(3, 8, 8, device=selected))
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(len(row["program"]), []).append(row)
    tensors = []
    for depth, values in sorted(grouped.items()):
        tensors.append(
            (
                torch.tensor([row["start"] for row in values], device=selected),
                torch.tensor([row["program"] for row in values], device=selected),
                torch.tensor([row["answer"] for row in values], device=selected),
                depth,
            )
        )
    first_loss = None
    final_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        transition = torch.softmax(logits, dim=-1)
        losses = []
        for starts, programs, answers, _depth in tensors:
            probabilities = _execute_tensor_programs(transition, starts, programs)
            losses.append(
                torch.nn.functional.nll_loss(probabilities.clamp_min(1e-12).log(), answers)
            )
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise R11Error("teacher capability learning became non-finite")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        first_loss = value if first_loss is None else first_loss
        final_loss = value
    transition = torch.softmax(logits.detach(), dim=-1).cpu().float().contiguous()
    return transition, {
        "steps": steps,
        "learning_rate": learning_rate,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "trainable_parameters": logits.numel(),
        "optimizer": "Adam",
    }


def transition_accuracy(transition: torch.Tensor, rows: Sequence[Mapping[str, Any]]) -> float:
    executor = RecurrentTransitionNeuralISA()
    with torch.inference_mode():
        probabilities = executor(transition, [str(row["prompt"]) for row in rows])
    predictions = probabilities.argmax(dim=-1).tolist()
    return sum(
        int(prediction == row["answer"]) for prediction, row in zip(predictions, rows)
    ) / len(rows)


class NativeResidualISAHost:
    """Frozen host whose native output head realizes neural-ISA residual state."""

    def __init__(
        self,
        host_key: str,
        *,
        codec: torch.Tensor,
        codec_sha256: str,
        scale: float,
        device: str,
    ) -> None:
        self.host_key = host_key
        self.host = R10FrozenNeuralHost(SPECS[host_key], device=device)
        self.device = self.host.device
        self.codec = codec.to(self.device).float().contiguous()
        if tuple(self.codec.shape) != (8, self.host.hidden_width):
            raise R11Error(f"host codec shape changed: {host_key}")
        if sha256_bytes(self.codec.cpu().numpy().tobytes()) != codec_sha256:
            raise R11Error(f"host codec identity changed: {host_key}")
        if not math.isfinite(scale) or scale <= 0:
            raise R11Error("host codec scale invalid")
        self.codec_sha256 = codec_sha256
        self.scale = float(scale)
        self.model_state_sha256 = self.host.model_state_sha256

    @property
    def target_token_ids(self) -> list[int]:
        return self.host.target_token_ids

    @torch.inference_mode()
    def base_state(self, prompts: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        base_logits, output = self.host.logits(prompts, prefix=None, output_hidden_states=True)
        if self.host.spec.encoder_decoder:
            hidden = output.decoder_hidden_states[-1][:, 0, :].float()
        else:
            encoded = self.host.encode(prompts)
            last = encoded["attention_mask"].sum(dim=1) - 1
            batch = torch.arange(len(prompts), device=self.device)
            hidden = output.hidden_states[-1][batch, last].float()
        return base_logits, hidden

    @torch.inference_mode()
    def realize(self, hidden: torch.Tensor, distributions: torch.Tensor) -> torch.Tensor:
        if distributions.shape != (hidden.shape[0], 8):
            raise R11Error("neural ISA distribution geometry changed")
        control = distributions.to(self.device).float() @ self.codec
        controlled = hidden + self.scale * control
        head = self.host.model.get_output_embeddings()
        logits = head(controlled.to(head.weight.dtype)).float()
        if not torch.isfinite(logits).all():
            raise R11Error("native recipient logits became non-finite")
        return logits

    @torch.inference_mode()
    def logits(self, prompts: Sequence[str], distributions: torch.Tensor | None) -> torch.Tensor:
        base_logits, hidden = self.base_state(prompts)
        return base_logits if distributions is None else self.realize(hidden, distributions)

    def canonical_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return self.host.canonical_probabilities(logits)

    def verify_frozen(self) -> None:
        self.host.verify_frozen()
        if module_sha256(self.host.model) != self.model_state_sha256:
            raise R11Error(f"recipient model changed: {self.host_key}")
        if sha256_bytes(self.codec.cpu().numpy().tobytes()) != self.codec_sha256:
            raise R11Error(f"recipient codec changed: {self.host_key}")
