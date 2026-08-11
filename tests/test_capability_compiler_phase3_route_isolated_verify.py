from abi.capability_compiler_phase3_route_isolated_verify import SYSTEMS


def test_verifier_system_matrix_is_locked():
    assert SYSTEMS == ("A0", "A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")
