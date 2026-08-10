from abi.capability_compiler_phase3_nonlinear_rank768_feasibility import accounting


def test_nonlinear_rank768_accounting() -> None:
    values = accounting()
    assert values["deployed_parameters"] == 399_903_744
    assert values["fp16_payload_bytes"] == 799_807_488
    assert values["active_incremental_macs_at_maximum_context"] == 307_540_992
    assert values["source_to_target_active_mac_ratio"] > 12
