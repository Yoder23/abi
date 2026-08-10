import torch

from abi.capability_compiler_phase3_identity_residual import ExactIdentityResidual, PARAMETERS


def test_identity_residual_budget_and_zero_near_initial_gate():
    bridge = ExactIdentityResidual()
    assert sum(value.numel() for value in bridge.parameters()) == PARAMETERS == 99_093
    hidden = torch.zeros(2, 768)
    prompt = torch.randn(7, 768)
    routes = torch.tensor([0, 9])
    adapted, attention = bridge.adapt(hidden, prompt, routes)
    assert adapted.shape == hidden.shape
    assert attention.shape == (2, 7)
    assert torch.all(bridge.copy_gate(hidden, routes) < 0.02)
