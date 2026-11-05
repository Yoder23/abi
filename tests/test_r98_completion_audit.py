from copy import deepcopy

import pytest

from experiments.plan_english_r98_completion import core as c
from scripts.audit_r98_completion import recompute


@pytest.fixture(scope="module")
def evidence():
    catalog, old = c.original_evidence()
    destination = c.OUT / "source"
    return (
        catalog,
        old,
        c.rows(destination / "completions.jsonl"),
        c.rows(destination / "rows.jsonl"),
        c.read(destination / "result.json"),
    )


def test_actual_completion_failure_recomputes(evidence):
    summary, gate, accounting = recompute(*evidence)
    assert summary["passing"] == 226
    assert summary["per_task"]["email_drafting"] == {"passing": 22, "rows": 30}
    assert gate is False
    assert accounting["total_source_requests"] == 271


@pytest.mark.parametrize(
    "attack",
    [
        "missing_completion",
        "duplicate_completion",
        "prefix",
        "omitted_merged",
        "output",
        "passing_boolean",
        "summary",
        "imported_tokens",
        "teacher_tokens",
        "time",
    ],
)
def test_negative_audit_rejects_hostile_changes(evidence, attack):
    catalog, old, completed, merged, result = deepcopy(evidence)
    if attack == "missing_completion":
        completed.pop()
    elif attack == "duplicate_completion":
        completed.append(completed[0])
    elif attack == "prefix":
        completed[0]["token_ids"][0] += 1
    elif attack == "omitted_merged":
        merged.pop()
    elif attack == "output":
        merged[0]["teacher_output"] += " tampered"
    elif attack == "passing_boolean":
        result["source_gate"] = True
    elif attack == "summary":
        result["summary"]["passing"] = 240
    elif attack == "imported_tokens":
        result["accounting"]["imported_completed_output_tokens"] = 1
    elif attack == "teacher_tokens":
        result["accounting"]["completion_acquisition"]["teacher_generated_tokens"] = 1
    else:
        result["accounting"]["completion_acquisition"]["source_inference_seconds"] = 0
    with pytest.raises(RuntimeError):
        recompute(catalog, old, completed, merged, result)
