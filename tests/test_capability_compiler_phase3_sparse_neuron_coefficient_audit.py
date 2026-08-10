import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_sparse_neuron_coefficient_audit import (
    FORMAT,
    _apply_correction,
    deployment_accounting,
)


def test_sparse_neuron_audit_format_is_versioned() -> None:
    assert "sparse-neuron-coefficient" in FORMAT
    assert FORMAT.endswith("/1")


def test_sparse_neuron_feature_accounting_is_exact() -> None:
    values = deployment_accounting(384)
    assert values["imported_sparse_feature_parameters_per_layer"] == 2_654_208
    assert values["source_blocks"] == 0


def test_sparse_correction_preserves_linear_when_zero() -> None:
    linear = torch.tensor([[1.0, 2.0]])
    sparse = torch.tensor([[3.0]])
    assert torch.equal(_apply_correction(linear, sparse, torch.zeros(1, 2)), linear)


def test_sparse_correction_rejects_shape_drift() -> None:
    with pytest.raises(Phase3Error, match="feature shape changed"):
        _apply_correction(torch.zeros(1, 2), torch.zeros(1, 3), torch.zeros(2, 2))
