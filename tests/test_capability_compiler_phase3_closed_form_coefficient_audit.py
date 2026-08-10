import torch

from abi.capability_compiler_phase3_closed_form_coefficient_audit import solve_ridge


def test_closed_form_map_recovers_linear_targets() -> None:
    torch.manual_seed(1)
    features = torch.randn(64, 5)
    expected = torch.randn(5, 3)
    targets = features @ expected
    actual, ridge = solve_ridge(features, targets, 1e-8)
    assert ridge > 0
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
