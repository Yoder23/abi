from abi.capability_compiler_phase3_native_causal_fit import _common_prefix


def test_common_prefix_is_exact() -> None:
    assert _common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert _common_prefix([1], [2]) == 0
