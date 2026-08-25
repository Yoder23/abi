from __future__ import annotations

from pathlib import Path

import pytest

from abi_v2.strict_validation import (
    StrictValidationError,
    read_jsonl,
    verify_certifications,
    verify_live_causality,
    verify_live_isolation,
    verify_locked_matrix_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_physical_certification_is_recomputed_from_raw_capsules() -> None:
    value = verify_certifications(ROOT)
    assert value["hosts_verified"] == 3
    assert value["physical_capability_archives_present"] == 0
    assert value["physical_source_success_ledgers_present"] == 0
    assert all(row["performance_observations"] >= 20 for row in value["hosts"].values())


def test_complete_locked_matrix_is_recomputed_from_all_raw_rows() -> None:
    value = verify_locked_matrix_rows(ROOT)
    assert value["rows_verified"] == 5043
    assert value["cross_host_outputs_equal"] == 1681
    assert value["cross_host_specialist_actions_equal"] == 300


def test_live_causality_is_recomputed_without_summary_flags() -> None:
    value = verify_live_causality(ROOT)
    assert value["raw_rows"] == 3072
    assert value["cross_host_real_outputs_equal"] == 128
    assert value["state_channel_supported"] is False


def test_live_isolation_is_recomputed_from_outputs_and_frozen_evaluators() -> None:
    value = verify_live_isolation(ROOT)
    assert value["raw_rows"] == 2100
    assert value["target_successes"] == 0
    assert value["cross_host_outputs_equal"] == 700


def test_jsonl_reader_fails_closed_on_missing_or_empty_raw_evidence(tmp_path: Path) -> None:
    with pytest.raises(StrictValidationError, match="unavailable"):
        read_jsonl(tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    with pytest.raises(StrictValidationError, match="empty"):
        read_jsonl(empty)


def test_jsonl_reader_fails_closed_on_blank_or_malformed_rows(tmp_path: Path) -> None:
    blank = tmp_path / "blank.jsonl"
    blank.write_bytes(b"{}\n\n")
    with pytest.raises(StrictValidationError, match="blank raw row"):
        read_jsonl(blank)
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"{not-json}\n")
    with pytest.raises(StrictValidationError, match="invalid raw row"):
        read_jsonl(malformed)
