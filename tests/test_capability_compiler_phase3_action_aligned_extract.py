import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_action_aligned_extract import _projection, _ragged_offsets, _tensor_hash


def test_projection_and_ragged_offsets_are_deterministic() -> None:
    first = _projection(12, 4, 66)
    second = _projection(12, 4, 66)
    assert torch.equal(first, second)
    assert _tensor_hash(first) == _tensor_hash(second)
    assert torch.equal(_ragged_offsets([2, 3, 1]), torch.tensor([0, 2, 5, 6]))
    with pytest.raises(Phase3Error):
        _ragged_offsets([1, 0])
