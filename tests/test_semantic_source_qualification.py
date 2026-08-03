from __future__ import annotations

import json

import pytest

from abi.semantic_source_qualification import (
    SemanticSourceQualificationError,
    _durable_full_generate,
    _judgment_passes,
    _parse_judgment,
    _preflight_capability_gates,
    _preflight_probe_ids,
    _qualification_summary,
)


def _judgment(**changes):
    value = {
        "linguistic_quality": 4,
        "prompt_grounding": 4,
        "task_correctness": 4,
        "instruction_adherence": 3,
        "changed_required_supplied_detail": False,
        "unsupplied_factual_detail": False,
        "repetition_or_collapse": False,
        "unusable_or_empty": False,
    }
    value.update(changes)
    return value


def test_semantic_judgment_parser_is_schema_closed():
    valid = _judgment()
    assert _parse_judgment(json.dumps(valid)) == valid
    assert _parse_judgment(json.dumps({**valid, "extra": 1})) is None
    assert _parse_judgment(json.dumps({**valid, "task_correctness": 5})) is None


def test_semantic_pass_requires_quality_grounding_correctness_and_no_flags():
    assert _judgment_passes(_judgment())
    assert _judgment_passes(_judgment(instruction_adherence=2))
    assert not _judgment_passes(_judgment(task_correctness=2))
    assert not _judgment_passes(
        _judgment(changed_required_supplied_detail=True)
    )


def test_preflight_selection_is_deterministic_and_stratifies_prior_results():
    rows = [
        {
            "probe_id": f"{capability}-{passed}-{index}",
            "capability": capability,
            "split": "search",
            "passed": passed,
        }
        for capability in ("grammar", "rewriting")
        for passed in (False, True)
        for index in range(3)
    ]
    selected = _preflight_probe_ids(
        rows, protocol_id="semantic-v1", per_capability=2
    )
    assert selected == _preflight_probe_ids(
        rows, protocol_id="semantic-v1", per_capability=2
    )
    assert len(selected) == 4
    for capability in ("grammar", "rewriting"):
        subset = [value for value in selected if value.startswith(capability)]
        assert any("-True-" in value for value in subset)
        assert any("-False-" in value for value in subset)


def test_full_summary_applies_search_depth_and_wilson_gate():
    observations = []
    for capability in ("grammar", "rewriting"):
        observations.extend(
            {
                "capability": capability,
                "split": "search",
                "passed": True,
            }
            for _ in range(100)
        )
        observations.extend(
            {
                "capability": capability,
                "split": "validation",
                "passed": True,
            }
            for _ in range(64)
        )
    summary = _qualification_summary(
        observations,
        minimum_pass_rate=0.9,
        minimum_wilson_lower_bound=0.8,
        minimum_search_passes=100,
    )
    assert summary["available_capabilities"] == 2
    assert all(row["available"] for row in summary["capabilities"].values())


def test_preflight_capability_gates_require_every_capability():
    summary = {
        "capabilities": {
            "conversation": {"search_passes": 19, "search_total": 20},
            "rewriting": {"search_passes": 18, "search_total": 20},
        }
    }
    assert _preflight_capability_gates(
        summary,
        {
            "minimum_semantic_passes_per_capability": 18,
            "minimum_semantic_pass_rate_per_capability": 0.9,
        },
    ) == {
        "minimum_semantic_passes_per_capability": True,
        "minimum_semantic_pass_rate_per_capability": True,
    }
    assert _preflight_capability_gates(
        summary,
        {"minimum_semantic_passes_per_capability": 19},
    ) == {"minimum_semantic_passes_per_capability": False}


class _FakeJudge:
    def __init__(self):
        self.calls = 0

    def generate_batch(self, requests):
        self.calls += 1
        return [
            {
                "output": "{}",
                "input_tokens": 4,
                "teacher_tokens": 2,
                "teacher_token_counter": "authoritative_generated_token_ids",
                "authoritative_generated_token_ids": [1, 2],
                "finish_reason": "eos_token",
                "generation_max_new_tokens": request["max_new_tokens"],
            }
            for request in requests
        ]


def test_durable_full_generation_resumes_without_repeating_completed_rows(tmp_path):
    requests = [
        {"prompt": f"judge {index}", "max_new_tokens": 8, "seed": 0}
        for index in range(3)
    ]
    ids = [f"probe-{index}" for index in range(3)]
    journal = tmp_path / "judge.partial.jsonl"
    first = _FakeJudge()
    generated, load_seconds, inference_seconds, evidence = _durable_full_generate(
        judge=first,
        requests=requests,
        selected_ids=ids,
        batch_size=2,
        journal_path=journal,
        journal_identity={"protocol_sha256": "a" * 64},
        current_load_seconds=1.5,
    )
    assert len(generated) == 3
    assert first.calls == 2
    assert load_seconds == 1.5
    assert inference_seconds >= 0
    assert evidence["completed_probes"] == 3

    resumed = _FakeJudge()
    generated_again, total_load, _, resumed_evidence = _durable_full_generate(
        judge=resumed,
        requests=requests,
        selected_ids=ids,
        batch_size=2,
        journal_path=journal,
        journal_identity={"protocol_sha256": "a" * 64},
        current_load_seconds=2.0,
    )
    assert generated_again == generated
    assert resumed.calls == 0
    assert total_load == 3.5
    assert resumed_evidence["sessions"] == 2

    journal.write_text(
        journal.read_text(encoding="utf-8").replace("probe-0", "probe-X", 1),
        encoding="utf-8",
    )
    with pytest.raises(SemanticSourceQualificationError, match="journal hash"):
        _durable_full_generate(
            judge=_FakeJudge(),
            requests=requests,
            selected_ids=ids,
            batch_size=2,
            journal_path=journal,
            journal_identity={"protocol_sha256": "a" * 64},
            current_load_seconds=1.0,
        )
