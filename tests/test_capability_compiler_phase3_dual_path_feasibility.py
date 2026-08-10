from abi.capability_compiler_phase3_dual_path_feasibility import dual_path_accounting


def test_dual_path_accounting_is_exact() -> None:
    value = dual_path_accounting(runtime_vocabulary=32_015, full_width=3_072, bottleneck_width=192, intermediate_size=768, layers=32, maximum_context=512, source_incremental_macs=3_823_042_560)
    assert value["copied_substrate_parameters"] == 196_899_840
    assert value["attention_parameters_per_layer"] == 1_327_296
    assert value["mlp_parameters_per_layer"] == 1_622_208
    assert value["trainable_parameters_per_layer"] == 2_949_504
    assert value["trainable_replacement_parameters"] == 94_384_128
    assert value["deployed_parameters"] == 291_283_968
    assert value["fp16_payload_bytes"] == 582_567_936
    assert value["target_incremental_macs_at_maximum_context"] == 199_013_376
    assert value["source_to_target_incremental_mac_ratio"] > 19.2
