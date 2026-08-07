import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_teacher_representation_extract import _find_unique
from abi.capability_compiler_phase3_teacher_representation_extract import (
    _find_unique_text,
    _offset_token_span,
)


def test_find_unique_requires_one_exact_span() -> None:
    assert _find_unique([9, 1, 2, 3, 8], [1, 2, 3]) == 1
    with pytest.raises(Phase3Error):
        _find_unique([1, 2, 1, 2], [1, 2])
    with pytest.raises(Phase3Error):
        _find_unique([1, 2], [3])


def test_find_unique_text_and_offset_span_require_exact_boundaries() -> None:
    text = "prefix\nFollow here.\nsuffix"
    start, end = _find_unique_text(text, "Follow here.")
    assert (start, end) == (7, 19)
    offsets = [(0, 6), (6, 7), (7, 8), (8, 13), (13, 18), (18, 19), (19, 20)]
    assert _offset_token_span(offsets, start, end) == (2, 4)
    with pytest.raises(Phase3Error):
        _find_unique_text("same same", "same")
    with pytest.raises(Phase3Error):
        _offset_token_span([(6, 10), (10, 19)], start, end)
