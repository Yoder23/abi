from abi import capability_compiler_phase4_functional_validation as subject


def test_split_is_deterministic_disjoint_and_exhaustive() -> None:
    ids = [f"record-{index}" for index in range(200)]
    validation = {value for value in ids if subject._partition(value, 5) == 0}
    train = set(ids) - validation
    assert train.isdisjoint(validation)
    assert train | validation == set(ids)
    assert validation
