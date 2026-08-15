from abi.capability_compiler_phase4_b20_host_compatibility_audit import host_can_change


def test_host_change_scope_is_declared_and_bounded():
    assert host_can_change({"capability": "coherence", "repetition_collapse_v2": False})
    assert host_can_change({"capability": "format_control", "repetition_collapse_v2": False})
    assert host_can_change({"capability": "clarification", "repetition_collapse_v2": False})
    assert host_can_change({"capability": "rewriting", "repetition_collapse_v2": True})
    assert not host_can_change({"capability": "fluent_realization", "repetition_collapse_v2": False})
