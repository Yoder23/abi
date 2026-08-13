from abi.capability_compiler_phase4_v19_frontier_rescreen import (
    _merged_evaluation,
    frontier_decision,
)


def test_merged_evaluation_counts_primary_fields():
    rows = [
        {
            "capability": capability,
            "functional_pass_v1": True,
            "functional_pass_v2": True,
            "repetition_collapse_v2": False,
            "capability_route_correct": True,
            "strong_parent_output_exact": capability not in {"abstention", "coherence", "fluent_realization", "tone_control"},
            "guard_terminated": False,
            "abstention_clause_prefixed": False,
        }
        for capability in (
            "grammar", "coherence", "prompt_grounding", "instruction_following",
            "conversation", "supplied_text_summarization", "rewriting",
            "email_drafting_from_notes", "tone_control", "format_control",
            "clarification", "abstention", "fact_free_reasoning", "fluent_realization",
        )
    ]
    result = _merged_evaluation(rows)
    assert result["functional_passes_v1"] == 14
    assert result["router_correct"] == 14
    assert result["strong_routes_exact"] == 10
    assert result["repetition_collapses_v2"] == 0


def test_frontier_requires_all_three_b80_pass_and_all_three_b40_fail():
    systems = [
        {"budget": "B40", "machine_gates_pass": False},
        {"budget": "B80", "machine_gates_pass": True},
        {"budget": "B40", "machine_gates_pass": False},
        {"budget": "B80", "machine_gates_pass": True},
        {"budget": "B40", "machine_gates_pass": False},
        {"budget": "B80", "machine_gates_pass": True},
    ]
    assert all(frontier_decision(systems).values())
    systems[0]["machine_gates_pass"] = True
    decision = frontier_decision(systems)
    assert decision["b80_all_seed_machine_gates"]
    assert not decision["b40_adjacent_lower_fails_all_seeds"]
