"""Differentiable finite-state source adapter for the R8 synthetic family."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .capability_generator import MODULUS, OPERATORS, canonical_json_bytes
from .native_host import FrozenNeuralHost, NativeHostError, tensor_state_sha256


class SourceTransitionError(RuntimeError):
    """Raised when the registered neural transition adapter changes schema."""


class NeuralTransitionSource(torch.nn.Module):
    """A learned stochastic transition system whose output enters source logits."""

    def __init__(self, host: FrozenNeuralHost, *, seed: int) -> None:
        super().__init__()
        self.host = host
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self.transition_logits = torch.nn.Parameter(
            0.01 * torch.randn(len(OPERATORS), MODULUS, MODULUS, generator=generator)
        )
        self.log_gain = torch.nn.Parameter(torch.tensor(1.0))
        self.digit_bias = torch.nn.Parameter(torch.tensor(1.0))
        self.initial_state = {
            name: value.detach().cpu().clone() for name, value in self.state_dict().items()
        }

    def reset(self) -> None:
        self.load_state_dict(self.initial_state, strict=True)

    def transition_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits.float(), dim=-1)

    def state_distribution(
        self,
        starts: Sequence[int],
        programs: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        if len(starts) != len(programs) or not starts:
            raise SourceTransitionError("structured source batch changed")
        device = self.transition_logits.device
        state = F.one_hot(
            torch.tensor(starts, dtype=torch.long, device=device),
            num_classes=MODULUS,
        ).float()
        matrices = self.transition_probabilities()
        maximum_depth = max(len(program) for program in programs)
        for depth in range(maximum_depth):
            active = torch.tensor(
                [depth < len(program) for program in programs],
                dtype=torch.bool,
                device=device,
            )
            operators = torch.tensor(
                [program[depth] if depth < len(program) else 0 for program in programs],
                dtype=torch.long,
                device=device,
            )
            selected = matrices.index_select(0, operators)
            updated = torch.einsum("bi,bij->bj", state, selected)
            state = torch.where(active.unsqueeze(1), updated, state)
        return state

    def canonical_addition(
        self,
        starts: Sequence[int],
        programs: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        state = self.state_distribution(starts, programs)
        gain = torch.exp(self.log_gain.float()).clamp(max=100.0)
        return self.digit_bias.float() + gain * torch.log(state.clamp_min(1e-12))

    def add_to_logits(
        self,
        base_logits: torch.Tensor,
        starts: Sequence[int],
        programs: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        addition = self.canonical_addition(starts, programs).to(base_logits)
        target_ids = torch.tensor(
            self.host.target_token_ids, dtype=torch.long, device=base_logits.device
        )
        delta = torch.zeros_like(base_logits)
        delta.index_copy_(-1, target_ids, addition)
        return base_logits + delta

    def logits(
        self,
        prompts: Sequence[str],
        starts: Sequence[int],
        programs: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        base, _ = self.host.logits(prompts, prefix=None)
        return self.add_to_logits(base, starts, programs)

    def state_sha256(self) -> str:
        return tensor_state_sha256(self.state_dict())

    def schema(self) -> dict[str, Any]:
        value = {
            "format": "abi-native-transfer-r8-neural-transition-source/1",
            "operators": len(OPERATORS),
            "states": MODULUS,
            "composition": "repeated stochastic-matrix multiplication",
            "source_logit_interface": "additive canonical-token logits",
            "public_structured_inputs": ["start_state", "opaque_operator_ids"],
            "parameters": sum(parameter.numel() for parameter in self.parameters()),
            "hidden_rule_inputs": 0,
        }
        value["schema_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
        return value


def controller_state(value: NeuralTransitionSource) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in value.state_dict().items()
    }


def load_controller_state(
    value: NeuralTransitionSource, state: Mapping[str, torch.Tensor]
) -> None:
    value.load_state_dict(dict(state), strict=True)
    if not torch.isfinite(value.transition_logits).all():
        raise SourceTransitionError("source transition state is non-finite")


def ensure_source_base_frozen(host: FrozenNeuralHost, expected_sha256: str) -> None:
    if any(parameter.requires_grad for parameter in host.model.parameters()):
        raise NativeHostError("source base parameter became trainable")
    if tensor_state_sha256(host.model.state_dict()) != expected_sha256:
        raise NativeHostError("source base state changed during transition training")


def pack_controller_states(
    states: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        f"capability_{index:03d}/{name}": tensor.detach().cpu().contiguous()
        for index, state in enumerate(states)
        for name, tensor in state.items()
    }


def unpack_controller_state(
    packed: Mapping[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    prefix = f"capability_{index:03d}/"
    result = {
        name.removeprefix(prefix): tensor
        for name, tensor in packed.items()
        if name.startswith(prefix)
    }
    if not result:
        raise SourceTransitionError(f"source transition state missing: {index}")
    return result
