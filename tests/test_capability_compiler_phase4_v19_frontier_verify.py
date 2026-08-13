from abi.capability_compiler_phase4_v19_frontier_verify import _pointer_checks, _without


def test_without_does_not_mutate_source():
    source = {"a": 1, "evidence_sha256": "x"}
    assert _without(source, "evidence_sha256") == {"a": 1}
    assert source["evidence_sha256"] == "x"


def test_pointer_checks_recompute_selection_and_rendering():
    def extract(_):
        return ("[A] first", "[B] second", "[C] third")

    def render(values):
        return "; ".join(values) + "."

    pointer = {
        "candidate_count": 6,
        "selected_index": 0,
        "candidate_token_lengths": [3] * 6,
        "model_log_probability_sums": [6, 5, 4, 3, 2, 1],
        "prompt_prefill_forward_passes": 1,
        "candidate_scoring_forward_passes": 1,
        "persistent_prompt_state_reused": True,
        "active_residual_routes": 1,
        "evaluator_used": False,
    }
    checks = _pointer_checks("prompt", "[A] first; [B] second; [C] third.", pointer, extract, render)
    assert all(checks.values())
