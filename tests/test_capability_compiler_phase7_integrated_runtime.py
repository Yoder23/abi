from abi.capability_compiler_phase7_integrated_runtime import (
    _domain_schedule,
    _selected_only,
)


def test_domain_runtime_schedule_has_100_distinct_plus_20_repeats():
    rows = [
        {"probe_id": f"{domain}-{index}", "domain": domain}
        for domain in ("chemistry", "civics", "python")
        for index in range(100)
    ]
    distinct, scheduled = _domain_schedule(rows)
    assert len(distinct) == 100
    assert len({row["probe_id"] for row in distinct}) == 100
    assert scheduled == [*distinct, *distinct[:20]]


def test_selected_only_rejects_inactive_work_and_missing_prefill():
    delta = {
        "chosen": {"prefill_calls": 1, "decode_step_calls": 3},
        "inactive": {"prefill_calls": 0, "decode_step_calls": 0},
    }
    assert _selected_only(delta, "chosen")
    delta["inactive"]["decode_step_calls"] = 1
    assert not _selected_only(delta, "chosen")
    delta["inactive"]["decode_step_calls"] = 0
    delta["chosen"]["prefill_calls"] = 0
    assert not _selected_only(delta, "chosen")
