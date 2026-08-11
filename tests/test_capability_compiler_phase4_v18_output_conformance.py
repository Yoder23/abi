from abi.capability_compiler_phase4_v18_output_conformance import compare_row


def test_compare_row_is_field_exact_and_fail_closed():
    row = {
        "probe_id": "p", "capability": "grammar", "output": "ok",
        "original_output": "ok", "output_token_ids": [1],
        "automatic_capability_route": "grammar", "control_residual_route": None,
        "task_route": 0, "guard_terminated": False,
    }
    assert all(compare_row(row, row).values())
    changed = dict(row, output="not ok")
    matches = compare_row(changed, row)
    assert matches["output"] is False
    assert sum(not value for value in matches.values()) == 1
