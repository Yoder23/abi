from experiments.preexisting_representation_r15b.public_qualification import (
    DEPTHS,
    OPERATIONS,
    apply_program,
    build_rows,
    wilson_lower,
)


def test_registered_operations_are_noncommutative_permutations() -> None:
    images = []
    for _name, multiplier, offset in OPERATIONS:
        image = tuple((multiplier * value + offset) % 8 for value in range(8))
        assert sorted(image) == list(range(8))
        images.append(image)
    left = apply_program(0, (0, 1))
    right = apply_program(0, (1, 0))
    assert left != right


def test_public_rows_are_deterministic_unique_and_correct() -> None:
    first = build_rows(rows_per_depth=32, seed=1515001)
    second = build_rows(rows_per_depth=32, seed=1515001)
    assert first == second
    assert len(first) == sum(min(32, 8 * 3**depth) for depth in DEPTHS)
    assert len({row["row_id"] for row in first}) == len(first)
    assert all(row["answer"] == apply_program(row["start"], tuple(row["program"])) for row in first)


def test_wilson_lower_is_fail_closed_and_monotonic() -> None:
    assert wilson_lower(0, 100) < 1e-15
    assert wilson_lower(80, 100) < wilson_lower(90, 100) < wilson_lower(100, 100)
