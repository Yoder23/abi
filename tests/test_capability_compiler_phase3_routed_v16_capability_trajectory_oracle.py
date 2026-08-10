from abi.capability_compiler_phase2_common import CAPABILITIES


def test_capability_oracle_has_fourteen_unique_routes():
    assert len(CAPABILITIES) == 14
    assert len(set(CAPABILITIES)) == 14
    assert "grammar" in CAPABILITIES
    assert "instruction_following" in CAPABILITIES
