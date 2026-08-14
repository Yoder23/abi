from abi.capability_compiler_phase4_v22_b40_clarification_attribution import (
    FORMAT,
    clarification_failure_taxonomy,
)


def test_attribution_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v22-b40-clarification-attribution/1"


def test_taxonomy_identifies_missing_interrogative_form():
    taxonomy = clarification_failure_taxonomy("I need additional information.")
    assert taxonomy == {
        "missing_question_mark": True,
        "missing_inquiry_marker": True,
        "empty_output": False,
    }


def test_taxonomy_accepts_valid_interrogative_form():
    taxonomy = clarification_failure_taxonomy("What should I improve?")
    assert not taxonomy["missing_question_mark"]
    assert not taxonomy["missing_inquiry_marker"]
    assert not taxonomy["empty_output"]


def test_taxonomy_keeps_orthogonal_failure_dimensions():
    question_without_marker = clarification_failure_taxonomy("More details?")
    assert not question_without_marker["missing_question_mark"]
    assert question_without_marker["missing_inquiry_marker"]
    marker_without_question = clarification_failure_taxonomy("Could you provide details.")
    assert marker_without_question["missing_question_mark"]
    assert not marker_without_question["missing_inquiry_marker"]
