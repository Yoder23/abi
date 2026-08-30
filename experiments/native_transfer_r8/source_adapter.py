"""Source-side LoRA capability adapters for the R8 teacher acquisition step."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch

from .native_host import NativeHostError, canonical_json_bytes, tensor_state_sha256


class SourceAdapterError(RuntimeError):
    """Raised when source LoRA installation or base-weight custody fails."""


class LoRAConv1D(torch.nn.Module):
    """LoRA for Hugging Face GPT-2 Conv1D without modifying its frozen base."""

    def __init__(self, base: torch.nn.Module, *, rank: int) -> None:
        super().__init__()
        if type(base).__name__ != "Conv1D" or not hasattr(base, "weight"):
            raise SourceAdapterError("source LoRA target is not a GPT-2 Conv1D")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        input_width, output_width = (int(value) for value in base.weight.shape)
        self.rank = int(rank)
        self.a = torch.nn.Parameter(torch.empty(input_width, self.rank))
        self.b = torch.nn.Parameter(torch.zeros(self.rank, output_width))
        self.reset()

    def reset(self) -> None:
        torch.nn.init.normal_(self.a, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        adapted = torch.matmul(torch.matmul(inputs.float(), self.a.float()), self.b.float())
        return base + adapted.to(base.dtype) / self.rank


class SourceLoRASet:
    """Installed source adapters plus exact base-weight custody checks."""

    def __init__(self, model: torch.nn.Module, *, rank: int, expected_base_sha256: str) -> None:
        self.model = model
        self.expected_base_sha256 = expected_base_sha256
        self.modules: dict[str, LoRAConv1D] = {}
        targets = [
            (name, module)
            for name, module in model.named_modules()
            if type(module).__name__ == "Conv1D"
        ]
        if len(targets) != 24:
            raise SourceAdapterError(f"expected 24 source Conv1D targets, found {len(targets)}")
        for name, module in targets:
            replacement = LoRAConv1D(module, rank=rank).to(module.weight.device)
            self._replace(name, replacement)
            self.modules[name] = replacement
        self.verify_base_frozen()

    def _replace(self, name: str, replacement: torch.nn.Module) -> None:
        parent = self.model
        pieces = name.split(".")
        for piece in pieces[:-1]:
            parent = getattr(parent, piece)
        setattr(parent, pieces[-1], replacement)

    def parameters(self) -> list[torch.nn.Parameter]:
        return [
            parameter
            for module in self.modules.values()
            for parameter in (module.a, module.b)
        ]

    def reset(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        for module in self.modules.values():
            module.reset()

    def state(self) -> dict[str, torch.Tensor]:
        result = {}
        for name, module in self.modules.items():
            result[f"{name}.lora_a"] = module.a.detach().cpu().contiguous()
            result[f"{name}.lora_b"] = module.b.detach().cpu().contiguous()
        return result

    def load_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {
            key for name in self.modules for key in (f"{name}.lora_a", f"{name}.lora_b")
        }
        if set(state) != expected:
            raise SourceAdapterError("source adapter tensor inventory changed")
        with torch.no_grad():
            for name, module in self.modules.items():
                module.a.copy_(state[f"{name}.lora_a"].to(module.a))
                module.b.copy_(state[f"{name}.lora_b"].to(module.b))

    def state_sha256(self) -> str:
        return tensor_state_sha256(self.state())

    def base_state_sha256(self) -> str:
        normalized = {}
        for name, value in self.model.state_dict().items():
            if any(name == f"{module_name}.{suffix}" for module_name in self.modules for suffix in ("a", "b")):
                continue
            normalized_name = name
            for module_name in self.modules:
                prefix = f"{module_name}.base."
                if name.startswith(prefix):
                    normalized_name = f"{module_name}." + name.removeprefix(prefix)
                    break
            normalized[normalized_name] = value
        return tensor_state_sha256(normalized)

    def verify_base_frozen(self) -> None:
        if self.base_state_sha256() != self.expected_base_sha256:
            raise SourceAdapterError("source base model changed after LoRA installation")
        for module in self.modules.values():
            if any(parameter.requires_grad for parameter in module.base.parameters()):
                raise SourceAdapterError("source base parameter became trainable")

    def inventory(self) -> dict[str, Any]:
        rows = [
            {
                "module": name,
                "rank": module.rank,
                "a_shape": list(module.a.shape),
                "b_shape": list(module.b.shape),
            }
            for name, module in sorted(self.modules.items())
        ]
        return {
            "modules": rows,
            "module_count": len(rows),
            "trainable_parameters": sum(value.numel() for value in self.parameters()),
            "inventory_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        }


def ensure_only_adapters_trainable(model: torch.nn.Module, adapters: SourceLoRASet) -> None:
    permitted = {id(value) for value in adapters.parameters()}
    for parameter in model.parameters():
        if parameter.requires_grad != (id(parameter) in permitted):
            raise NativeHostError("source trainable-parameter boundary changed")


def pack_capability_states(
    states: list[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    return {
        f"capability_{index:03d}/{name}": value.detach().cpu().contiguous()
        for index, state in enumerate(states)
        for name, value in state.items()
    }


def unpack_capability_state(
    packed: Mapping[str, torch.Tensor], index: int
) -> dict[str, torch.Tensor]:
    prefix = f"capability_{index:03d}/"
    result = {
        name.removeprefix(prefix): value
        for name, value in packed.items()
        if name.startswith(prefix)
    }
    if not result:
        raise SourceAdapterError(f"source adapter capability state missing: {index}")
    return result
