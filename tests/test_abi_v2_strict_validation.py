from __future__ import annotations

from pathlib import Path

import pytest
import torch

from abi_v2.live_causality import _native_parameter_intervention
from abi_v2.strict_validation import (
    StrictValidationError,
    read_jsonl,
    verify,
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
    assert value["applied_host_state_channel"].startswith("AppliedHostStateAdapter")


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


def test_top_level_verifier_fails_closed_on_unrecomputable_tree(tmp_path: Path) -> None:
    with pytest.raises(StrictValidationError, match="unavailable"):
        verify(tmp_path)


def test_random_native_intervention_is_deterministic_and_recomputable() -> None:
    first = torch.nn.Linear(40, 40)
    second = torch.nn.Linear(40, 40)
    second.load_state_dict(first.state_dict())
    left = _native_parameter_intervention(
        first, condition="random_state", host_key="qwen2"
    )
    right = _native_parameter_intervention(
        second, condition="random_state", host_key="qwen2"
    )
    assert left == right
    assert left["kind"] == "live_native_parameter_intervention"
    assert left["values_after"] != left["values_before"]


def test_host_removed_intervention_contains_no_native_tensor() -> None:
    value = _native_parameter_intervention(
        None, condition="host_removed", host_key="qwen2"
    )
    assert value["kind"] == "structural_native_host_absence"
    assert value["parameter_name"] is None
    assert value["values_before"] == value["values_after"] == []
