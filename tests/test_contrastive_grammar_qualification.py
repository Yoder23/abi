from __future__ import annotations

from abi.contrastive_grammar_qualification import (
    _aggregate,
    _choice_prompt,
    _observation_from_scores,
    _pair_from_probe,
)
from abi.natural_grammar_reference_catalog import (
    build_natural_grammar_preflight_catalog,
)


def _score(request_id: str, value: float, label: str) -> dict:
    return {
        "request_id": request_id,
        "sum_log_probability": value,
        "completion_token_ids": [ord(label)],
        "rendered_prompt_sha256": "a" * 64,
    }


def test_pair_and_counterbalanced_prompts_preserve_both_sentences():
    probe = build_natural_grammar_preflight_catalog()["probes"][0]
    wrong, correct = _pair_from_probe(probe)
    assert wrong != correct
    first = _choice_prompt(wrong, correct)
    swapped = _choice_prompt(correct, wrong)
    assert f"A: {wrong}" in first and f"B: {correct}" in first
    assert f"A: {correct}" in swapped and f"B: {wrong}" in swapped


def test_observation_requires_correct_preference_in_both_orders():
    probe = build_natural_grammar_preflight_catalog()["probes"][0]
    probe_id = probe["probe_id"]
    wrong, correct = _pair_from_probe(probe)
    passing_scores = {
        f"{probe_id}:ab:correct": _score("ab-c", -0.1, "B"),
        f"{probe_id}:ab:incorrect": _score("ab-i", -2.0, "A"),
        f"{probe_id}:ba:correct": _score("ba-c", -0.2, "A"),
        f"{probe_id}:ba:incorrect": _score("ba-i", -1.0, "B"),
    }
    observation = _observation_from_scores(
        probe=probe, wrong=wrong, correct=correct, scores=passing_scores
    )
    assert observation["passed"] is True
    assert observation["ab"]["margin"] > 0
    assert observation["ba"]["margin"] > 0
    failing_scores = dict(passing_scores)
    failing_scores[f"{probe_id}:ba:correct"] = _score("ba-c", -3.0, "A")
    failed = _observation_from_scores(
        probe=probe, wrong=wrong, correct=correct, scores=failing_scores
    )
    assert failed["passed"] is False


def test_aggregate_enforces_global_and_structure_gates():
    catalog = build_natural_grammar_preflight_catalog()
    observations = []
    for probe in catalog["probes"]:
        probe_id = probe["probe_id"]
        wrong, correct = _pair_from_probe(probe)
        scores = {
            f"{probe_id}:ab:correct": _score("ab-c", -0.1, "B"),
            f"{probe_id}:ab:incorrect": _score("ab-i", -2.0, "A"),
            f"{probe_id}:ba:correct": _score("ba-c", -0.2, "A"),
            f"{probe_id}:ba:incorrect": _score("ba-i", -1.0, "B"),
        }
        observations.append(
            _observation_from_scores(
                probe=probe, wrong=wrong, correct=correct, scores=scores
            )
        )
    summary, checks, failures = _aggregate(
        observations,
        expected_records=16,
        minimum_total_passes=14,
        minimum_search_passes=0,
        minimum_validation_pass_rate=0.0,
        minimum_validation_wilson_lower_bound=0.0,
        minimum_search_passes_per_structure=1,
        minimum_validation_passes_per_structure=0,
    )
    assert summary["passes"] == 16
    assert all(checks.values())
    assert failures == []
