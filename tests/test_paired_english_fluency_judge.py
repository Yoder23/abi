from __future__ import annotations

from abi.paired_english_fluency_judge import (
    _bootstrap_ratio,
    _candidate_is_a,
    _parse_scores,
    _selected_probe_ids,
)


def test_judge_parser_is_schema_closed() -> None:
    valid = {
        side: {
            "grammatical_fluency": 4,
            "local_and_global_coherence": 3,
            "prompt_and_context_grounding": 4,
            "instruction_and_format_adherence": 3,
            "unsupported_factual_detail": False,
            "repetition_or_collapse": False,
            "unusable_or_empty": False,
        }
        for side in ("A", "B")
    }
    import json

    assert _parse_scores(json.dumps(valid)) == valid
    invalid = dict(valid)
    invalid["A"] = dict(invalid["A"], grammatical_fluency=5)
    assert _parse_scores(json.dumps(invalid)) is None


def test_judge_orientation_is_deterministic() -> None:
    assert _candidate_is_a("probe-1") == _candidate_is_a("probe-1")
    assert _candidate_is_a(
        "probe-1", "validation-protocol"
    ) == _candidate_is_a("probe-1", "validation-protocol")


def test_paired_bootstrap_reports_exact_equal_ratio() -> None:
    result = _bootstrap_ratio([(12.0, 12.0), (8.0, 8.0)])
    assert result["lower_95"] == 1.0
    assert result["median"] == 1.0
    assert result["upper_95"] == 1.0


def test_judge_selection_can_be_preregistered_on_validation() -> None:
    catalog = {
        "probes": [
            {
                "split": split,
                "capability": capability,
                "probe_id": f"{split}-{capability}-{index}",
            }
            for split in ("validation", "final_test")
            for capability in ("rewriting", "coherence")
            for index in range(4)
        ]
    }
    selected = _selected_probe_ids(
        catalog,
        split="validation",
        protocol_id="validation-protocol",
        prompts_per_capability=2,
    )
    assert len(selected) == 4
    assert all(value.startswith("validation-") for value in selected)
