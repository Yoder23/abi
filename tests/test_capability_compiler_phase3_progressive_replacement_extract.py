from abi.capability_compiler_phase3_progressive_replacement_extract import (
    runtime_source_rows,
    substrate_parameter_count,
)


def test_progressive_replacement_source_row_mapping_is_collision_free_at_host_boundary() -> None:
    rows = runtime_source_rows(
        external_actions=32_011,
        host_special_source_rows=[32_000, 32_001, 32_007, 0],
    )
    assert len(rows) == 32_015
    assert len(set(rows)) == 32_011
    assert rows[:4] == [32_000, 32_001, 32_007, 0]
    assert rows[4] == 0 and rows[-1] == 32_010


def test_progressive_replacement_copied_substrate_count_is_exact() -> None:
    assert substrate_parameter_count(runtime_vocabulary=32_015, full_width=3_072, layers=32) == 196_899_840
