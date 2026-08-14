from abi.capability_compiler_phase4_b50_baselines import (
    EXACT_B50_ROUTER_MEMBERSHIPS,
    FORMAT,
    configuration_allowed,
    stage_authorizes_system,
    validate_exact_b50_router_records,
)
from abi.capability_compiler_phase3 import Phase3Error
import pytest


PROTOCOL = {
    "stages": {"grid": {"authorized_systems": ["L0", "D0"]}},
    "systems": {
        "L0": {
            "ranks": [16, 64],
            "learning_rates": [0.0001, 0.0003],
            "exposures": [1, 4],
        },
        "D0": {"learning_rates": [0.00001, 0.00003], "exposures": [1, 2, 4]},
    }
}


def test_baseline_format_is_versioned():
    assert FORMAT == "abi-capability-compiler-phase4-b50-baselines/1"


def test_configuration_validation_preserves_frozen_grids():
    assert configuration_allowed(
        PROTOCOL,
        system="L0",
        rank=16,
        learning_rate=0.0001,
        exposures=4,
    )
    assert not configuration_allowed(
        PROTOCOL,
        system="L0",
        rank=32,
        learning_rate=0.0001,
        exposures=4,
    )
    assert configuration_allowed(
        PROTOCOL,
        system="D0",
        rank=None,
        learning_rate=0.00003,
        exposures=2,
    )


def test_stage_system_authorization_fails_closed():
    assert stage_authorizes_system(PROTOCOL, stage="grid", system="L0")
    assert not stage_authorizes_system(PROTOCOL, stage="grid", system="D1")
    assert not stage_authorizes_system(PROTOCOL, stage="headline", system="L0")


def test_exact_b50_router_membership_depth_is_frozen():
    rows = [
        {"capability": capability}
        for capability, count in EXACT_B50_ROUTER_MEMBERSHIPS.items()
        for _ in range(count)
    ]
    validate_exact_b50_router_records(rows)
    with pytest.raises(Phase3Error, match="membership depth changed"):
        validate_exact_b50_router_records(rows[:-1])
