import torch

from abi.capability_compiler_phase3_route_isolated import CONTROL_SYSTEMS, PARAMETERS, RANK, RouteIsolatedResidual


def test_route_isolated_parameter_and_shape_contract():
    model = RouteIsolatedResidual()
    assert sum(value.numel() for value in model.parameters()) == PARAMETERS
    hidden = torch.randn(4, 3, 768)
    result = model.delta(hidden, torch.arange(4))
    assert result.shape == hidden.shape
    assert RANK == 16


def test_legacy_rank64_partition_is_exactly_four_disjoint_slices():
    model = RouteIsolatedResidual()
    state = {"norm.weight": torch.ones(768), "norm.bias": torch.zeros(768), "down.weight": torch.arange(64 * 768).reshape(64, 768).float(), "up.weight": torch.arange(768 * 64).reshape(768, 64).float(), "route_scale.weight": torch.zeros(4, 64), "route_shift.weight": torch.zeros(4, 64)}
    model.load_state_dict(state)
    assert torch.equal(model.down[2], state["down.weight"][32:48])
    assert torch.equal(model.up[3], state["up.weight"][:, 48:64])


def test_control_matrix_is_locked():
    assert CONTROL_SYSTEMS == ("A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")
