from abi.capability_compiler_phase3_progressive_replacement_feasibility import replacement_parameter_accounting
from abi.capability_compiler_phase3_progressive_replacement_feasibility_repair import corrected_accounting


def test_corrected_progressive_replacement_accounting_uses_runtime_actions() -> None:
    original = replacement_parameter_accounting(
        vocabulary=32_064,
        full_width=3_072,
        bottleneck_width=192,
        intermediate_size=768,
        layers=32,
        maximum_context=512,
    )
    value = corrected_accounting(original, runtime_vocabulary=32_015, source_vocabulary=32_064, full_width=3_072)
    assert value["runtime_vocabulary"] == 32_015
    assert value["omitted_unused_source_rows"] == 49
    assert value["copied_source_parameters"] == 196_899_840
    assert value["deployed_parameters"] == 253_535_232
    assert value["fp16_payload_bytes"] == 507_070_464
    assert value["target_incremental_macs_at_maximum_context"] == 161_264_640
    assert value["source_to_target_incremental_mac_ratio"] > 23.7
