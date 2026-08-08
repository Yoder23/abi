from abi.capability_compiler_phase3_acquisition_coverage import _ngrams


def test_ngrams_are_ordered_and_exact() -> None:
    assert list(_ngrams([1, 2, 3], 2)) == [(1, 2), (2, 3)]
    assert list(_ngrams([1], 2)) == []
