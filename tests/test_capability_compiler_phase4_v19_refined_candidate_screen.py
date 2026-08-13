from abi.capability_compiler_phase2_common import CAPABILITIES
from abi.capability_compiler_phase4_v19_frontier_rescreen import _merged_evaluation


def test_noncoherence_collapse_survives_coherence_replacement():
    rows = [{
        "capability": capability,
        "functional_pass_v1": True,
        "functional_pass_v2": True,
        "repetition_collapse_v2": capability == "supplied_text_summarization",
        "capability_route_correct": True,
        "strong_parent_output_exact": capability not in {"abstention", "coherence", "fluent_realization", "tone_control"},
        "guard_terminated": False,
        "abstention_clause_prefixed": False,
    } for capability in CAPABILITIES]
    result = _merged_evaluation(rows)
    assert result["repetition_collapses_v2"] == 1
