import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_b40_baselines import (
    CLI_STAGES,
    EXACT_B40_ROUTER_MEMBERSHIPS,
    FORMAT,
    configuration_allowed,
    stage_authorizes_system,
    validate_exact_b40_router_records,
)


PROTOCOL = {
    "stages": {"headline": {"authorized_systems": ["L0", "L1", "D0"]}},
    "systems": {
        "L0": {"ranks": [16], "learning_rates": [0.0001], "exposures": [4]},
        "L1": {"ranks": [8], "learning_rates": [0.0001], "exposures": [4]},
        "D0": {"learning_rates": [0.00003], "exposures": [4]},
    },
}


def test_b40_baseline_contract_is_budget_specific():
    assert FORMAT == "abi-capability-compiler-phase4-b40-baselines/1"
    assert CLI_STAGES == ("headline",)


def test_only_frozen_configuration_is_allowed():
    assert configuration_allowed(PROTOCOL, system="L0", rank=16, learning_rate=0.0001, exposures=4)
    assert not configuration_allowed(PROTOCOL, system="L0", rank=64, learning_rate=0.0001, exposures=4)
    assert configuration_allowed(PROTOCOL, system="D0", rank=None, learning_rate=0.00003, exposures=4)


def test_headline_authorization_fails_closed():
    assert stage_authorizes_system(PROTOCOL, stage="headline", system="L1")
    assert not stage_authorizes_system(PROTOCOL, stage="grid", system="L1")


def test_exact_b40_router_membership_depth_is_frozen():
    rows = [
        {"capability": capability}
        for capability, count in EXACT_B40_ROUTER_MEMBERSHIPS.items()
        for _ in range(count)
    ]
    validate_exact_b40_router_records(rows)
    with pytest.raises(Phase3Error, match="membership depth changed"):
        validate_exact_b40_router_records(rows[:-1])
