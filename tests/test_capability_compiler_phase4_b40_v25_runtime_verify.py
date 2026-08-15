from abi.capability_compiler_phase4_b40_v25_runtime_verify import (
    _cross_device_identity,
    _unique_schedule,
)


def test_cross_device_identity_requires_prompt_output_and_tokens():
    row = {"probe_id": "p", "output": "x", "output_token_ids": [1]}
    assert _cross_device_identity([row], [dict(row)]) == 1
    assert _cross_device_identity([row], [{**row, "output": "y"}]) == 0
    assert _cross_device_identity([row], [{**row, "output_token_ids": [2]}]) == 0


def test_schedule_requires_120_rows_and_100_distinct_prompts():
    rows = [{"probe_id": str(index % 100)} for index in range(120)]
    assert _unique_schedule(rows)
    assert not _unique_schedule(rows[:-1])
    mutated = [dict(row) for row in rows]
    mutated[99]["probe_id"] = mutated[0]["probe_id"]
    assert not _unique_schedule(mutated)
