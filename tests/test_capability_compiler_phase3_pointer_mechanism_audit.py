from abi.capability_compiler_phase3_pointer_mechanism_audit import _quantile


def test_quantile_is_deterministic_and_bounded():
    values = [0.9, 0.1, 0.5, 0.3, 0.7]
    assert _quantile(values, 0.0) == 0.1
    assert _quantile(values, 0.5) == 0.5
    assert _quantile(values, 1.0) == 0.9
    assert _quantile([], 0.95) == 0.0
