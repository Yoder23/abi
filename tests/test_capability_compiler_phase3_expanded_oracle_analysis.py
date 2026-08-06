import copy

import pytest

from abi.capability_compiler_phase3_expanded_oracle_analysis import (
    ExpandedOracleAnalysisError,
    adversarial_checks,
    verify_decision,
)


def test_decision_verifier_fails_closed():
    expected = {"status": "failed", "phase3_certified": False}
    changed = copy.deepcopy(expected)
    changed["phase3_certified"] = True
    with pytest.raises(ExpandedOracleAnalysisError):
        verify_decision(changed, expected)


def test_adversarial_mutations_are_rejected():
    value = {
        "functional_passes": 1248,
        "repetition_collapses": 75,
        "decision": {"capacity_sufficient": False},
        "phase3_certified": False,
        "abi_superiority_claim_allowed": False,
        "attribution": {"sealed_layercake_regression": False},
    }
    result = adversarial_checks(value)
    assert result["status"] == "PASS"
    assert result["count"] == 6
