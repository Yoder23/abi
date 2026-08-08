import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_native_causal_extract import _predecessor_indices


def test_predecessor_indices_are_causal_and_complete() -> None:
    assert _predecessor_indices(5, 3) == [4, 5, 6]
    with pytest.raises(Phase3Error):
        _predecessor_indices(0, 3)
    with pytest.raises(Phase3Error):
        _predecessor_indices(5, 0)
