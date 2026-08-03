from __future__ import annotations

import pytest

from abi.training_sequence_coverage_audit import (
    TrainingSequenceCoverageAuditError,
    _summarize_lengths,
)


def test_sequence_coverage_audit_counts_omitted_targets_by_capability():
    rows = [
        {
            "record_id": "one",
            "capability": "coherence",
            "prompt_tokens": 5,
            "response_tokens": 4,
        },
        {
            "record_id": "two",
            "capability": "coherence",
            "prompt_tokens": 8,
            "response_tokens": 7,
        },
        {
            "record_id": "three",
            "capability": "rewriting",
            "prompt_tokens": 3,
            "response_tokens": 2,
        },
    ]
    summary = _summarize_lengths(rows, sequence_ceiling=10)
    assert summary["rows"] == 3
    assert summary["prompt_at_or_above_ceiling"] == 0
    assert summary["full_sequence_above_ceiling"] == 1
    assert summary["omitted_teacher_target_tokens"] == 5
    assert summary["by_capability"]["coherence"] == {
        "rows": 2,
        "truncated_rows": 1,
        "omitted_teacher_target_tokens": 5,
    }
    assert summary["maximum_row"]["record_id"] == "two"


def test_sequence_coverage_audit_rejects_empty_or_invalid_ceiling():
    with pytest.raises(TrainingSequenceCoverageAuditError):
        _summarize_lengths([], sequence_ceiling=10)
    with pytest.raises(TrainingSequenceCoverageAuditError):
        _summarize_lengths(
            [
                {
                    "record_id": "one",
                    "capability": "coherence",
                    "prompt_tokens": 1,
                    "response_tokens": 1,
                }
            ],
            sequence_ceiling=0,
        )
