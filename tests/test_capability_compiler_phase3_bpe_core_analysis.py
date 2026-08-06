import pytest

from abi.capability_compiler_phase3_bpe_core_analysis import (
    paired_stratified_bootstrap,
    wilson,
)
from abi.capability_compiler_phase2_common import CAPABILITIES


def test_wilson_rejects_invalid_counts():
    with pytest.raises(Exception):
        wilson(2, 1)


def test_paired_bootstrap_is_deterministic_and_stratified():
    rows = []
    for capability in CAPABILITIES:
        rows.extend(
            {
                "capability": capability,
                "candidate_pass": index < 90,
                "teacher_pass": index < 95,
            }
            for index in range(100)
        )
    first = paired_stratified_bootstrap(rows, replicates=100, seed=38)
    second = paired_stratified_bootstrap(rows, replicates=100, seed=38)
    assert first == second
    assert first["candidate_minus_teacher"] == pytest.approx(-0.05)
