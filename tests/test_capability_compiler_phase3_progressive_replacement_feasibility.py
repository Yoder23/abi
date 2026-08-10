from abi.capability_compiler_phase3_progressive_replacement_feasibility import (
    calibration_cache_accounting,
    replacement_parameter_accounting,
)


def test_progressive_replacement_parameter_accounting_is_exact() -> None:
    accounting = replacement_parameter_accounting(
        vocabulary=32_064,
        full_width=3_072,
        bottleneck_width=192,
        intermediate_size=768,
        layers=32,
        maximum_context=512,
    )
    assert accounting["input_embedding_parameters"] == 98_500_608
    assert accounting["output_head_parameters"] == 98_500_608
    assert accounting["copied_source_norm_parameters"] == 196_608
    assert accounting["copied_final_norm_parameters"] == 3_072
    assert accounting["projection_parameters_per_layer"] == 1_179_648
    assert accounting["attention_parameters_per_layer"] == 147_456
    assert accounting["mlp_parameters_per_layer"] == 442_368
    assert accounting["latent_norm_parameters_per_layer"] == 384
    assert accounting["trained_parameters_per_layer"] == 1_769_856
    assert accounting["trained_replacement_parameters"] == 56_635_392
    assert accounting["copied_source_parameters"] == 197_200_896
    assert accounting["deployed_parameters"] == 253_836_288
    assert accounting["fp16_payload_bytes"] == 507_672_576
    assert accounting["maximum_context_kv_cache_bytes_fp16"] == 12_582_912
    assert accounting["source_to_target_incremental_mac_ratio"] > 23.0


def test_progressive_replacement_calibration_cache_is_bounded() -> None:
    assert calibration_cache_accounting(tokens=65_536, full_width=3_072) == {
        "calibration_tokens": 65_536,
        "one_fp16_hidden_field_bytes": 402_653_184,
        "input_and_target_fp16_hidden_bytes": 805_306_368,
    }
