from __future__ import annotations

import pytest
import torch

from abi.layercake_direct_source_initialization import (
    DirectSourceInitializationError,
    copy_source_substrate,
    parse_selected_layers,
)


class _Transformer(torch.nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.wte = torch.nn.Embedding(11, 4)
        self.wpe = torch.nn.Embedding(8, 4)
        self.h = torch.nn.ModuleList(
            [torch.nn.Linear(4, 4) for _ in range(layers)]
        )
        self.ln_f = torch.nn.LayerNorm(4)


class _Core(torch.nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.transformer = _Transformer(layers)
        self.task_classifier = torch.nn.Linear(4, 2)
        self.task_cakes = torch.nn.ModuleList([torch.nn.Linear(4, 4)])


def test_direct_source_copy_is_exact_and_sparse_components_are_unchanged():
    torch.manual_seed(3)
    source = _Core(6)
    torch.manual_seed(7)
    target = _Core(3)
    cakes_before = {
        name: value.detach().clone()
        for name, value in target.task_cakes.state_dict().items()
    }
    classifier_before = {
        name: value.detach().clone()
        for name, value in target.task_classifier.state_dict().items()
    }

    result = copy_source_substrate(
        target=target, source=source, selected_layers=(1, 3, 5)
    )

    assert result["selected_source_layer_indices"] == [1, 3, 5]
    assert result["source_transformer_blocks_retained_exact_at_initialization"] == 3
    assert result["source_parameters_copied_at_initialization"] > 0
    assert torch.equal(target.transformer.wte.weight, source.transformer.wte.weight)
    for target_index, source_index in enumerate((1, 3, 5)):
        for name, value in target.transformer.h[target_index].state_dict().items():
            assert torch.equal(value, source.transformer.h[source_index].state_dict()[name])
    for name, value in cakes_before.items():
        assert torch.equal(value, target.task_cakes.state_dict()[name])
    for name, value in classifier_before.items():
        assert torch.equal(value, target.task_classifier.state_dict()[name])


def test_selected_source_layers_are_fail_closed():
    assert parse_selected_layers("1,3,5", source_layers=6, target_layers=3) == (
        1,
        3,
        5,
    )
    with pytest.raises(DirectSourceInitializationError):
        parse_selected_layers("1,1,5", source_layers=6, target_layers=3)
    with pytest.raises(DirectSourceInitializationError):
        parse_selected_layers("0,2", source_layers=6, target_layers=3)
