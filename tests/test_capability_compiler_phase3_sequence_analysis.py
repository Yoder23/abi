from abi.capability_compiler_phase3_sequence_analysis import compute_gates


def _system(passes=96, collapses=0):
    return {
        "repetition_collapses": collapses,
        "per_capability": {
            name: {
                "wilson": {"point": passes / 100, "lower_95": 0.91}
            }
            for name in (
                "grammar",
                "coherence",
                "fluent_realization",
                "prompt_grounding",
                "instruction_following",
                "conversation",
                "clarification",
                "abstention",
                "supplied_text_summarization",
                "rewriting",
                "email_drafting_from_notes",
                "tone_control",
                "format_control",
                "fact_free_reasoning",
            )
        },
    }


def _requirements():
    return {
        "per_capability_functional_point_estimate_minimum": 0.9,
        "per_capability_functional_wilson_lower_minimum": 0.85,
        "prompt_grounding_instruction_adherence_and_abstention_point_minimum": 0.95,
        "prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum": 0.9,
        "teacher_relative_paired_difference_lower_bound_minimum": -0.05,
        "repetition_collapse_count_maximum": 0,
        "B0_minus_each_B1_B2_B3_B4_paired_functional_bootstrap_lower_minimum": 0.0,
    }


def test_initial_gate_requires_quality_causality_teacher_and_zero_collapse():
    systems = {name: _system() for name in ("B0", "B1", "B2", "B3", "B4")}
    comparisons = {
        "B0_minus_B1": {"lower_95": 0.1},
        "B0_minus_B2": {"lower_95": 0.1},
        "B0_minus_B3": {"lower_95": 0.1},
        "B0_minus_B4": {"lower_95": 0.1},
        "B0_minus_T0": {"lower_95": -0.01},
    }
    gates = compute_gates(systems, comparisons, _requirements())
    assert all(gates.values())

    systems["B0"]["repetition_collapses"] = 1
    assert not compute_gates(systems, comparisons, _requirements())[
        "zero_repetition_collapses"
    ]
    systems["B0"]["repetition_collapses"] = 0
    comparisons["B0_minus_B4"]["lower_95"] = 0.0
    assert not compute_gates(systems, comparisons, _requirements())[
        "causal_B0_beats_each_control"
    ]
