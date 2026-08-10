from abi.capability_compiler_phase3_routed_v15_layer2_neuron_identity_audit import FORMAT


def test_routed_v15_neuron_identity_format_is_versioned() -> None:
    assert "neuron-identity-audit" in FORMAT
    assert FORMAT.endswith("/1")
