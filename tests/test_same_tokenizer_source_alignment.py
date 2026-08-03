from __future__ import annotations

import torch

from abi.same_tokenizer_source_alignment import (
    install_source_lora,
    merge_source_lora,
)


class _Config:
    n_layer = 1


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = torch.nn.Module()
        self.attn.c_attn = torch.nn.Linear(4, 4, bias=False)
        self.attn.c_proj = torch.nn.Linear(4, 4, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.c_fc = torch.nn.Linear(4, 4, bias=False)
        self.mlp.c_proj = torch.nn.Linear(4, 4, bias=False)


class _TinySource(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _Config()
        self.h = torch.nn.ModuleList([_Block()])


def test_source_lora_is_temporary_and_merges_into_same_shapes() -> None:
    model = _TinySource()
    before_shapes = {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    }
    parameters, names = install_source_lora(model, rank=2, alpha=2.0)
    assert len(names) == 4
    assert sum(parameter.numel() for parameter in parameters) == 64
    for parameter in parameters:
        parameter.data.add_(0.1)

    merge_source_lora(model)

    assert all("parametrizations" not in name for name in model.state_dict())
    assert {
        name: tuple(value.shape) for name, value in model.state_dict().items()
    } == before_shapes
