from abi.capability_compiler_phase4_b50_l1_conformance import (
    CRITICAL_CAPABILITIES,
    FORMAT,
    REPLACEMENT_PHRASE,
    SEEDS,
    SOURCE_PHRASE,
    _absolute_gates,
    _quality_report,
    conform_output,
)


def test_contract_constants_are_narrow():
    assert FORMAT == "abi-capability-compiler-phase4-b50-l1-conformance/1"
    assert SEEDS == (104729, 130363, 155921)
    assert CRITICAL_CAPABILITIES == (
        "prompt_grounding",
        "instruction_following",
        "abstention",
    )
    assert SOURCE_PHRASE == "cannot be known"
    assert REPLACEMENT_PHRASE == "is unknown"


def test_conformance_is_capability_bounded_and_single_replacement():
    original = "The answer cannot be known from the information given."
    changed, count = conform_output(original, "abstention")
    assert changed == "The answer is unknown from the information given."
    assert count == 1
    unchanged, count = conform_output(original, "grammar")
    assert unchanged == original
    assert count == 0


def test_conformance_is_case_insensitive_but_does_not_match_nearby_text():
    changed, count = conform_output("It CANNOT BE KNOWN today.", "abstention")
    assert changed == "It is unknown today."
    assert count == 1
    unchanged, count = conform_output("It cannot be inferred today.", "abstention")
    assert unchanged == "It cannot be inferred today."
    assert count == 0


def test_quality_gates_fail_closed():
    capabilities = (
        "grammar",
        "coherence",
        "prompt_grounding",
        "instruction_following",
        "conversation",
        "supplied_text_summarization",
        "rewriting",
        "email_drafting_from_notes",
        "tone_control",
        "format_control",
        "clarification",
        "abstention",
        "fact_free_reasoning",
        "fluent_realization",
    )
    rows = [
        {
            "probe_id": f"{capability}-{index}",
            "capability": capability,
            "functional_pass_v1": True,
            "functional_pass_v2": True,
            "repetition_collapse_v2": False,
        }
        for capability in capabilities
        for index in range(100)
    ]
    thresholds = {
        "per_capability_functional_point_estimate_minimum": 0.9,
        "per_capability_functional_wilson_lower_minimum": 0.85,
        "critical_point_minimum": 0.95,
        "critical_wilson_lower_minimum": 0.9,
        "repetition_collapse_v2_count_maximum": 0,
    }
    assert all(_absolute_gates(_quality_report(rows), thresholds).values())
    rows[1100]["functional_pass_v1"] = False
    rows[1100]["repetition_collapse_v2"] = True
    gates = _absolute_gates(_quality_report(rows), thresholds)
    assert not gates["zero_repetition_collapse"]
