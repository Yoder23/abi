import torch

from abi.capability_compiler_phase3_conditional_support_capacity_oracle import (
    marginal_reduction_scores,
    select_contributions,
)


def test_marginal_scores_equal_direct_squared_error_reduction():
    target = torch.tensor([[2.0, -1.0]])
    contributions = torch.tensor([[[1.0, 0.0], [0.0, -2.0], [-1.0, 0.0]]])
    scores = marginal_reduction_scores(target, contributions)
    direct = torch.stack(
        [target.square().sum(-1) - (target - contributions[:, index]).square().sum(-1) for index in range(3)],
        dim=-1,
    )
    torch.testing.assert_close(scores, direct)


def test_selection_is_stable_fixed_count_and_unscaled():
    target = torch.tensor([[2.0, -1.0]])
    contributions = torch.tensor([[[1.0, 0.0], [0.0, -2.0], [-1.0, 0.0]]])
    selected, indices = select_contributions(target, contributions, 2)
    assert indices.tolist() == [[0, 1]]
    torch.testing.assert_close(selected, torch.tensor([[1.0, -2.0]]))
