from abi.capability_compiler_phase4_v19_frontier_rescreen import _merged_evaluation


def test_noncoherence_collapse_survives_coherence_replacement():
    row = {
        "capability": "supplied_text_summarization",
        "functional_pass_v1": True,
        "functional_pass_v2": True,
        "repetition_collapse_v2": True,
        "capability_route_correct": True,
        "strong_parent_output_exact": True,
        "guard_terminated": False,
        "abstention_clause_prefixed": False,
    }
    result = _merged_evaluation([row])
    assert result["repetition_collapses_v2"] == 1
