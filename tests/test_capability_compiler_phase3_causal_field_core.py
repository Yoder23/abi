import torch

from abi.capability_compiler_phase3_causal_field_core import _losses, _map_external


def test_terminal_and_external_action_mapping_is_collision_free():
    assert _map_external(32007, 32007) == 2
    assert _map_external(0, 32007) == 4
    assert _map_external(32006, 32007) == 32010


def test_grouped_soft_loss_rewards_selected_and_residual_mass():
    logits = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 4.0, 2.0]]])
    positions = torch.tensor([[0]])
    targets = torch.tensor([[4]])
    field_ids = torch.tensor([[[4]]])
    field_probabilities = torch.tensor([[[0.75]]])
    field_residual = torch.tensor([[0.25]])
    valid = torch.tensor([[True]])
    hard, soft = _losses(logits, positions, targets, field_ids, field_probabilities, field_residual, valid)
    assert torch.isfinite(hard) and torch.isfinite(soft)
    assert hard > 0 and soft > 0
