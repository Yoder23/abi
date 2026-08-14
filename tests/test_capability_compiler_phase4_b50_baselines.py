from abi.capability_compiler_phase4_b50_baselines import (
    FORMAT,
    configuration_allowed,
    stage_authorizes_system,
)


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
