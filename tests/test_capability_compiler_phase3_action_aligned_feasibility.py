import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_action_aligned_feasibility import _overlaps, _piece_spans


def test_piece_spans_and_overlap_mapping_fail_closed() -> None:
    assert _piece_spans("hello", [b"he", b"llo"], 3) == [(3, 5), (5, 8)]
    assert _overlaps([(0, 2), (2, 5), (5, 8)], 3, 7) == [1, 2]
    with pytest.raises(Phase3Error):
        _piece_spans("hello", [b"hell"])
    with pytest.raises(Phase3Error):
        _overlaps([(0, 2)], 2, 4)
