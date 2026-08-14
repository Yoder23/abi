from abi.capability_compiler_phase4_v21_b50_format_attribution import (
    FORMAT,
    classify_delta,
    edit_distance,
)


def test_format_attribution_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v21-b50-format-attribution/1"


def test_edit_distance_known_cases():
    assert edit_distance("", "abc") == 3
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("same", "same") == 0


def test_delta_classifies_exact_and_prefix_boundaries():
    assert classify_delta("a\nb", "a\nb")["primary"] == "exact"
    assert classify_delta("a", "abc")["primary"] == "truncated_canonical_prefix"
    assert classify_delta("abc!", "abc")["primary"] == "extra_suffix"
    assert classify_delta("!abc", "abc")["primary"] == "extra_prefix"


def test_delta_classifies_whitespace_case_structure_and_lexical():
    assert (
        classify_delta("name:  Jon\ncount: 5", "name: Jon\ncount: 5")["primary"]
        == "whitespace_or_linebreak_only"
    )
    assert classify_delta("NAME: JON", "name: jon")["primary"] == "case_only"
    assert classify_delta("name: Jon count: 5", "name: Jon\ncount: 5")["primary"] == "line_structure_mismatch"
    assert classify_delta("name: Ana", "name: Jon")["primary"] == "lexical_or_identifier_mismatch"


def test_delta_reports_token_recall_and_lengths():
    result = classify_delta("code: N390", "code: N390MIRA")
    assert 0.0 < result["reference_token_recall"] < 1.0
    assert result["candidate_characters"] < result["reference_characters"]
