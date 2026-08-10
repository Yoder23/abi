from abi.capability_compiler_phase3_source_neuron_subset_audit import deployment_accounting

def test_nested_source_neuron_accounting() -> None:
    assert deployment_accounting(384)["source_to_target_active_mac_ratio"] > deployment_accounting(1536)["source_to_target_active_mac_ratio"]
    assert deployment_accounting(1536)["source_to_target_active_mac_ratio"] > 4
