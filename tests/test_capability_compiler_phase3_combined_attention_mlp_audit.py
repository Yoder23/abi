import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_combined_attention_mlp_audit import FORMAT, _reconstruct


def test_combined_attention_mlp_audit_format_is_versioned() -> None:
    assert "combined-attention-mlp-audit" in FORMAT
    assert FORMAT.endswith("/1")


def test_reconstruct_applies_coefficient_map_and_basis() -> None:
    feature = torch.tensor([[2.0, 3.0]])
    mean = torch.tensor([1.0, -1.0])
    weights = torch.eye(2)
    basis = torch.eye(2)
    assert torch.equal(_reconstruct(feature, mean, basis, weights), torch.tensor([[3.0, 2.0]]))


def test_reconstruct_rejects_shape_drift() -> None:
    with pytest.raises(Phase3Error, match="map shape changed"):
        _reconstruct(torch.zeros(1, 3), torch.zeros(2), torch.eye(2), torch.eye(2))
