import torch

from abi.capability_compiler_phase3_routed_v16_trajectory_retargeting import (
    HOST_EOS,
    solve_from_moments,
    source_token_id,
)


def test_source_token_mapping_is_exact():
    assert source_token_id(HOST_EOS, 32000) == 32000
    assert source_token_id(4, 32000) == 0
    assert source_token_id(103, 32000) == 99


def test_streamed_ridge_recovers_fixed_linear_map():
    torch.manual_seed(7)
    features = torch.randn(256, 12)
    expected = torch.randn(12, 9)
    targets = features @ expected
    solution, ridge = solve_from_moments(features.T @ features, features.T @ targets, 1e-8)
    assert ridge > 0
    assert torch.allclose(features @ solution, targets, atol=2e-5, rtol=2e-5)
