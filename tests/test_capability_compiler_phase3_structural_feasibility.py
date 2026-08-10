from abi.capability_compiler_phase3_structural_feasibility import target_parameter_accounting


def test_structural_target_parameter_accounting_is_exact() -> None:
    accounting = target_parameter_accounting(
        external_actions=32_011,
        host_special_actions=4,
        width=192,
        intermediate_size=768,
        layers=4,
    )
    assert accounting == {
        "vocabulary": 32_015,
        "lexical_parameters": 12_293_760,
        "attention_parameters_per_layer": 147_456,
        "mlp_parameters_per_layer": 442_368,
        "norm_parameters_per_layer": 384,
        "body_parameters": 2_360_832,
        "final_norm_parameters": 192,
        "deployed_parameters": 14_654_784,
        "source_derived_parameters": 14_653_248,
        "host_initialized_special_parameters": 1_536,
        "fp16_payload_bytes": 29_309_568,
    }
