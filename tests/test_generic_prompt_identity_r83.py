from __future__ import annotations

import torch

from abi.layercake_core_loader import ABIEnglishCoreConfig
from experiments.generic_prompt_identity_r83.extension_v1 import (
    BASE_ARCHITECTURE,
    install_generic_prompt_identity,
)


def test_r83_additive_pointer_preserves_every_parent_tensor() -> None:
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.wte = torch.nn.Embedding(32, 768)
    model.config = ABIEnglishCoreConfig(
        vocab_size=32,
        layers=6,
        architecture_version=BASE_ARCHITECTURE,
    )
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    install_generic_prompt_identity(model, initialize=True, selective=True)
    assert model.config.architecture_version == BASE_ARCHITECTURE
    assert model._abi_prompt_identity_bridge is model.prompt_identity
    assert not getattr(model, "_abi_task_route_layerwise_control", False)
    assert sum(parameter.numel() for parameter in model.prompt_identity.parameters()) == 49_931
    for name, expected in before.items():
        assert torch.equal(model.state_dict()[name], expected)
