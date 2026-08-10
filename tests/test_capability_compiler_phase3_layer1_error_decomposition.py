from abi.capability_compiler_phase3_layer1_error_decomposition import classify_bottleneck


def test_decomposition_classification_is_fail_closed() -> None:
    assert classify_bottleneck(
        exact_residual_pass=False, rank192_oracle_pass=False,
        maximum_rank_oracle_pass=True, maximum_rank_map_pass=True,
    ) == "ATTENTION_CAPACITY_OR_OPTIMIZATION_PRIMARY"
    assert classify_bottleneck(
        exact_residual_pass=True, rank192_oracle_pass=False,
        maximum_rank_oracle_pass=True, maximum_rank_map_pass=True,
    ) == "MLP_RESIDUAL_RANK_PRIMARY"
    assert classify_bottleneck(
        exact_residual_pass=True, rank192_oracle_pass=True,
        maximum_rank_oracle_pass=True, maximum_rank_map_pass=False,
    ) == "COEFFICIENT_MAP_PRIMARY"
