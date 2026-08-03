import pytest

from abi.source_runtime_evidence_audit import (
    SourceRuntimeEvidenceAuditError,
    _summarize_runtime_records,
)


def _record(identifier, capability, *, finish="eos_token"):
    return {
        "record_id": identifier,
        "capability": capability,
        "teacher_tokens": 3,
        "authoritative_generated_token_ids": [7, 8, 9],
        "finish_reason": finish,
        "generation_max_new_tokens": 3,
        "teacher_input_tokens": 5,
    }


def test_runtime_summary_keeps_functional_failures_separate_from_transport():
    records = [_record("one", "grammar"), _record("two", "rewriting")]
    results = [
        {"record_id": "one", "passed": True},
        {"record_id": "two", "passed": False},
    ]
    summary = _summarize_runtime_records(records, results)
    assert summary["runtime_evidence_complete"] == 2
    assert summary["eos_terminated"] == 2
    assert summary["length_terminated"] == 0
    assert summary["passing_probe_results"] == 1
    assert summary["failing_probe_results"] == 1


def test_runtime_summary_reports_length_termination_explicitly():
    summary = _summarize_runtime_records(
        [_record("one", "grammar", finish="length")],
        [{"record_id": "one", "passed": False}],
    )
    assert summary["length_terminated"] == 1
    assert summary["eos_terminated"] == 0
