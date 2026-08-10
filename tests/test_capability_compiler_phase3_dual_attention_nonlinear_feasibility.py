from abi.capability_compiler_phase3_dual_attention_nonlinear_feasibility import accounting


def test_dual_attention_nonlinear_accounting_is_exact() -> None:
    values = accounting()
    assert values["deployed_parameters"] == 517_874_688
    assert values["fp16_payload_bytes"] == 1_035_749_376
    assert values["active_incremental_macs_at_maximum_context"] == 425_505_792
    assert values["source_to_target_active_mac_ratio"] > 8.9
