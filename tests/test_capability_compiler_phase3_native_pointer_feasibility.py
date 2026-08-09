import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_native_pointer_feasibility import pointer_decode, pointer_encode


def test_native_pointer_round_trip_prefers_unique_source_actions() -> None:
    source = [10, 11, 12, 11]
    target = [10, 11, 12, 2]
    encoded, pointers = pointer_encode(source, target, 100)
    assert encoded == [100, 11, 102, 2]
    assert pointers == 2
    assert pointer_decode(encoded, source, 100) == target


def test_native_pointer_decode_rejects_out_of_range_position() -> None:
    with pytest.raises(Phase3Error, match="outside"):
        pointer_decode([105], [10], 100)
