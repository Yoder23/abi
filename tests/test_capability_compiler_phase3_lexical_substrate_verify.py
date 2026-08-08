from abi.capability_compiler_phase3_lexical_substrate_verify import selected_rows


def test_selected_rows_are_stable_unique_and_bounded():
    rows = selected_rows("input", 8, 32011)
    assert rows == selected_rows("input", 8, 32011)
    assert len(rows) == len(set(rows)) == 8
    assert min(rows) >= 0 and max(rows) < 32011
