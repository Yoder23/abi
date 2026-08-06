from abi.capability_compiler_phase3_shared_output_analysis import compute_gates


def _system(passes=96, collapses=0):
    names = (
        "grammar", "coherence", "fluent_realization", "prompt_grounding",
        "instruction_following", "conversation", "clarification", "abstention",
        "supplied_text_summarization", "rewriting", "email_drafting_from_notes",
        "tone_control", "format_control", "fact_free_reasoning",
    )
    return {"repetition_collapses": collapses, "per_capability": {name: {"wilson": {"point": passes / 100, "lower_95": 0.91}} for name in names}}


def test_shared_output_gate_requires_all_quality_and_causal_families():
    systems = {name: _system() for name in ("C0", "C1", "C2", "C3", "C4")}
    comparisons = {f"C0_minus_{name}": {"lower_95": 0.1} for name in ("C1", "C2", "C3", "C4")}
    comparisons["C0_minus_T0"] = {"lower_95": -0.01}
    requirements = {
        "per_capability_functional_point_estimate_minimum": 0.9,
        "per_capability_functional_wilson_lower_minimum": 0.85,
        "prompt_grounding_instruction_adherence_and_abstention_point_minimum": 0.95,
        "prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum": 0.9,
        "teacher_relative_paired_difference_lower_bound_minimum": -0.05,
        "repetition_collapse_count_maximum": 0,
        "C0_minus_each_C1_C2_C3_C4_paired_functional_bootstrap_lower_minimum": 0.0,
    }
    assert all(compute_gates(systems, comparisons, requirements).values())
    comparisons["C0_minus_C1"]["lower_95"] = 0.0
    assert not compute_gates(systems, comparisons, requirements)["causal_C0_beats_each_control"]
