from pathlib import Path

import torch

from abi import capability_compiler_phase4_capability_isolated_adaptation as subject


def test_capability_isolated_residual_has_one_physical_active_path() -> None:
    residual = subject.CapabilityIsolatedResidual()
    assert sum(value.numel() for value in residual.parameters()) == subject.PARAMETERS == 365_568
    assert subject.PARAMETERS_PER_ROUTE == 26_112
    hidden = torch.randn(2, 4, subject.WIDTH)
    with torch.no_grad():
        residual.up.zero_()
        residual.up[3].fill_(0.01)
    output = residual.delta(hidden, torch.tensor([3, 4]))
    assert output[0].abs().sum() > 0
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_protocol_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = root / "ABI_CAPABILITY_COMPILER_PHASE4_CAPABILITY_ISOLATED_EVALUATION_REPAIR_V669.json"
    if protocol.exists():
        result = subject.preflight(root, protocol)
        assert result["status"] == "PASS_CAPABILITY_ISOLATED_ADAPTATION_PREFLIGHT"
        assert result["one_active_path_verified"] is True
        assert result["final_test_accessed"] is False
