import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_residual_nonlinear_rank768_fit import (
    FORMAT,
    _combined_coefficients,
)


def test_residual_nonlinear_format_is_versioned() -> None:
    assert "residual-nonlinear-rank768" in FORMAT
    assert FORMAT.endswith("/1")


def test_zero_nonlinear_correction_preserves_linear_coefficients() -> None:
    linear = torch.tensor([[1.0, 2.0]])
    assert torch.equal(_combined_coefficients(linear, torch.zeros_like(linear)), linear)


def test_combined_coefficients_reject_shape_drift() -> None:
    with pytest.raises(Phase3Error, match="shape changed"):
        _combined_coefficients(torch.zeros(1, 2), torch.zeros(1, 3))
