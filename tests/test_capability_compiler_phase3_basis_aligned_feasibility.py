from abi.capability_compiler_phase3_basis_aligned_feasibility import accounting


def test_basis_aligned_accounting_keeps_rank_and_active_compute() -> None:
    value = accounting(
        copied_substrate=196_899_840,
        current_dual_trainable=94_384_128,
        full_width=3_072,
        rank=192,
        layers=32,
        current_target_macs=199_013_376,
        source_macs=3_823_042_560,
    )
    assert value["imported_mlp_basis_and_mean_parameters"] == 18_972_672
    assert value["trainable_coefficient_and_attention_parameters"] == 75_509_760
    assert value["deployed_parameters"] == 291_382_272
    assert value["fp16_payload_bytes"] == 582_764_544
    assert value["active_incremental_macs_at_maximum_context"] == 199_013_376
