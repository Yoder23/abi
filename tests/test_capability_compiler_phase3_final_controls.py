from abi.capability_compiler_phase3_final_controls import (
    SYSTEMS,
    _derange_targets,
    _prompt_hash_route,
)


def _row(index: int):
    return {
        "record_id": f"r{index:02d}",
        "capability": "coherence",
        "builder": 0,
        "view": "host_projected",
        "input_ids": [10, index + 20, 99, index + 40],
        "labels": [-100, -100, 99, index + 40],
        "prompt_tokens": 2,
        "response_tokens": 2,
    }


def test_prompt_hash_route_is_deterministic_and_label_independent():
    row = _row(0)
    changed = dict(row, capability="tone_control")
    assert _prompt_hash_route(row) == _prompt_hash_route(changed)
    assert 0 <= _prompt_hash_route(row) < 4


def test_derangement_has_no_identity_and_preserves_prompts():
    rows = [_row(index) for index in range(80)]
    changed = _derange_targets(rows)
    assert len(changed) == 80
    assert all(row["deranged_target_record_id"] != row["record_id"] for row in changed)
    assert all(row["input_ids"][:2] == original["input_ids"][:2] for row, original in zip(changed, rows))
    assert all(row["input_ids"][2:] != original["input_ids"][2:] for row, original in zip(changed, rows))


def test_derangement_skips_an_offset_with_duplicate_response_content():
    rows = [_row(index) for index in range(80)]
    for index in range(0, 80, 2):
        rows[index + 1]["input_ids"][2:] = rows[index]["input_ids"][2:]
        rows[index + 1]["labels"][2:] = rows[index]["labels"][2:]
    changed = _derange_targets(rows)
    assert all(row["input_ids"][2:] != original["input_ids"][2:] for row, original in zip(changed, rows))
    assert len({row["derangement_offset"] for row in changed}) == 1


def test_system_matrix_is_locked():
    assert SYSTEMS == ("A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")
