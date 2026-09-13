from __future__ import annotations

import torch

from abi.layercake_core_loader import (
    ABIEnglishCoreConfig,
    CAPABILITY_CAKE_CANONICAL_ROUTES,
    CAPABILITY_CAKE_ORDER,
    SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
)
from experiments.mixture_pointer_r84.extension_v1 import install_mixture_pointer


def test_r84_pointer_preserves_deep_adapter_parent_tensors() -> None:
    model = torch.nn.Module()
    model.transformer = torch.nn.Module()
    model.transformer.wte = torch.nn.Embedding(32, 768)
    model.config = ABIEnglishCoreConfig(
        vocab_size=32,
        layers=6,
        task_cakes=14,
        capability_cake_order=CAPABILITY_CAKE_ORDER,
        capability_cake_canonical_routes=CAPABILITY_CAKE_CANONICAL_ROUTES,
        capability_router_buckets=4096,
        capability_router_width=32,
        capability_adapter_rank=32,
        architecture_version=SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
    )
    model._abi_deep_capability_adapters = True
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    install_mixture_pointer(model, initialize=True, selective=False)
    assert sum(parameter.numel() for parameter in model.prompt_identity.parameters()) == 49_935
    assert model.config.architecture_version == SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE
    for name, expected in before.items():
        assert torch.equal(model.state_dict()[name], expected)
