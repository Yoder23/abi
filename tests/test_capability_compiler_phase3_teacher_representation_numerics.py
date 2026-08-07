import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_teacher_representation_numerics import _batch_groups


def test_batch_groups_reconstruct_original_batch_starts() -> None:
    assert _batch_groups([0, 7, 8, 19, 31], 8) == [0, 8, 16, 24]
    with pytest.raises(Phase3Error):
        _batch_groups([1], 0)
