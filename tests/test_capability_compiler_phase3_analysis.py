import pytest

from abi.capability_compiler_phase3_analysis import (
    Phase3AnalysisError,
    measured_bottleneck,
    require_identical_successful_sequences,
    stratified_bootstrap,
    wilson,
)
from abi.capability_compiler_phase2_common import CAPABILITIES


def test_wilson_interval_is_ordered_and_contains_point():
    interval = wilson(90, 100)
    assert interval["lower_95"] < interval["point"] < interval["upper_95"]


def test_stratified_bootstrap_rejects_unpaired_and_preserves_clear_win():
    left = {}
    right = {}
    for capability in CAPABILITIES:
        for index in range(100):
            probe = f"{capability}-{index}"
            left[probe] = {"capability": capability, "functional_pass": True}
            right[probe] = {"capability": capability, "functional_pass": index >= 20}
    result = stratified_bootstrap(left, right, replicates=1000, seed=1729)
    assert result["lower_95"] > 0
    right.pop(next(iter(right)))
    with pytest.raises(Phase3AnalysisError, match="identities"):
        stratified_bootstrap(left, right, replicates=1000, seed=1729)


def test_measured_bottleneck_is_derived_from_current_evidence():
    system = {
        "trainable_parameters": 606_730,
        "functional_passes": 379,
        "repetition_collapses": 150,
        "per_capability": {
            "one": {"passes": 0, "observations": 100},
            "two": {"passes": 0, "observations": 100},
            "three": {"passes": 7, "observations": 100},
        },
    }
    text = measured_bottleneck(system)
    assert "606,730-parameter" in text
    assert "2 capabilities scored 0/100" in text
    assert "379/300" in text
    assert "150 outputs collapsed" in text


def test_successful_sequence_guard_rejects_mismatch_and_missing_hash():
    systems = {
        system: {"successful_record_sequence_sha256": "a" * 64}
        for system in ("A0", "A1", "A2", "A3", "A4")
    }
    assert require_identical_successful_sequences(systems) == "a" * 64

    systems["A4"]["successful_record_sequence_sha256"] = "b" * 64
    with pytest.raises(Phase3AnalysisError, match="sequences differ"):
        require_identical_successful_sequences(systems)

    systems["A4"]["successful_record_sequence_sha256"] = None
    with pytest.raises(Phase3AnalysisError, match="sequences differ"):
        require_identical_successful_sequences(systems)
