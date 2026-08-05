import pytest

from abi.capability_compiler_phase3_analysis import Phase3AnalysisError, stratified_bootstrap, wilson
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
