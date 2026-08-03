from __future__ import annotations

import pytest

from abi.layercake_full_core_acquisition import (
    FullCoreAcquisitionError,
    _apply_target_control,
)
from abi.grammar_transfer_pilot import paired_bootstrap_interval


def _rows() -> list[dict[str, str]]:
    return [
        {"record_id": f"{index:064x}", "response": response}
        for index, response in enumerate(("alpha", "beta", "gamma", "delta"), 1)
    ]


def test_target_derangement_is_deterministic_complete_and_non_mutating() -> None:
    original = _rows()
    first, first_contract = _apply_target_control(
        original,
        mode="deterministic_derangement",
        seed="pilot-control",
    )
    second, second_contract = _apply_target_control(
        original,
        mode="deterministic_derangement",
        seed="pilot-control",
    )
    original_by_id = {row["record_id"]: row["response"] for row in original}
    assert first == second
    assert first_contract == second_contract
    assert first_contract["all_targets_changed"] is True
    assert first_contract["mapping_count"] == len(original)
    assert all(
        row["response"] != original_by_id[row["record_id"]]
        for row in first
    )
    assert original == _rows()


def test_target_control_identity_and_invalid_pairing() -> None:
    rows = _rows()
    controlled, contract = _apply_target_control(
        rows, mode="identity", seed=""
    )
    assert controlled == rows
    assert controlled is not rows
    assert contract["mapping_sha256"] is None
    with pytest.raises(FullCoreAcquisitionError, match="at least two"):
        _apply_target_control(
            rows[:1],
            mode="deterministic_derangement",
            seed="pilot-control",
        )


def test_paired_bootstrap_is_deterministic_and_positive() -> None:
    first = paired_bootstrap_interval(
        [0.5, 1.0, 1.5, 2.0], replicates=1000, seed=7
    )
    second = paired_bootstrap_interval(
        [0.5, 1.0, 1.5, 2.0], replicates=1000, seed=7
    )
    assert first == second
    assert first["mean"] == 1.25
    assert first["lower_95"] > 0.0
