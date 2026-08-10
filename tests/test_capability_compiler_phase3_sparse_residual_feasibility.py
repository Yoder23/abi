from abi.capability_compiler_phase3_sparse_residual_feasibility import accounting


def test_sparse_residual_accounting_and_top1_active_cost() -> None:
    values = accounting()
    assert values["union_rank"] == 768
    assert values["deployed_parameters"] == 391_154_688
    assert values["fp16_payload_bytes"] == 782_309_376
    assert values["active_incremental_macs_at_maximum_context"] == 185_250_816
    assert values["active_mac_increase_over_direct_linear"] < 0.02
