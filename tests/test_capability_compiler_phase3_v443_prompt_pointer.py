import torch

from abi.capability_compiler_phase3_v443_prompt_pointer import (
    BRIDGE_PARAMETERS,
    BRIDGE_RANK,
    BRIDGE_ROUTES,
    _bridge,
)


def test_bridge_shape_and_parameter_budget_are_fixed():
    bridge = _bridge(torch.device("cpu"))
    assert bridge.rank == BRIDGE_RANK == 32
    assert bridge.route_bias.num_embeddings == BRIDGE_ROUTES == 10
    assert sum(value.numel() for value in bridge.parameters()) == BRIDGE_PARAMETERS == 49_931


def test_bridge_starts_with_sparse_copy_gate():
    bridge = _bridge(torch.device("cpu"))
    hidden = torch.zeros(3, 768)
    routes = torch.tensor([0, 4, 9])
    assert torch.all(bridge.copy_gate(hidden, routes) < 0.02)
