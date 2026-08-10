from abi.capability_compiler_phase3_weak_support_audit import _family, _ngrams


def test_family_is_derived_from_locked_probe_index():
    assert _family("phase1-validation-coherence-0013-v3") == "index_mod4_1"
    assert _family("phase1-validation-tone_control-0035-v2") == "index_mod4_3"


def test_ngrams_are_contiguous_and_bounded():
    assert list(_ngrams([1, 2, 3, 4], 3)) == [(1, 2, 3), (2, 3, 4)]
    assert list(_ngrams([1], 2)) == []
