from abi.capability_compiler_phase4_capability_bank_feasibility import _fold


def test_fold_is_deterministic_and_bounded():
    values = [_fold(f"probe-{index}", 5) for index in range(100)]
    assert values == [_fold(f"probe-{index}", 5) for index in range(100)]
    assert set(values) == {0, 1, 2, 3, 4}
