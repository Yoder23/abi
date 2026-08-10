import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_residual_attention_fit import FORMAT, _combine_attention


def test_residual_attention_format_is_local_and_versioned() -> None:
    assert "residual-attention" in FORMAT
    assert FORMAT.endswith("/1")


def test_zero_secondary_delta_preserves_primary_exactly() -> None:
    hidden = torch.tensor([[[1.0, 2.0]]])
    primary = torch.tensor([[[3.0, 5.0]]])
    assert torch.equal(_combine_attention(primary, hidden, hidden), primary)


def test_residual_attention_rejects_shape_drift() -> None:
    with pytest.raises(Phase3Error, match="shape changed"):
        _combine_attention(torch.zeros(1, 1, 2), torch.zeros(1, 2, 2), torch.zeros(1, 1, 2))
