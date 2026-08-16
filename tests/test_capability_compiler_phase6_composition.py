from abi.capability_compiler_phase6_composition import (
    DOMAINS,
    SEEDS,
    _selected_only,
)


def test_phase6_matrix_is_paired_and_domain_complete():
    assert SEEDS == (104729, 130363, 155921)
    assert DOMAINS == ("chemistry", "civics", "python")


def test_selected_only_requires_one_selected_prefill_and_zero_inactive_work():
    delta = {
        "chemistry": {
            "module_load_calls": 1,
            "prefill_calls": 1,
            "decode_step_calls": 10,
        },
        "python": {
            "module_load_calls": 0,
            "prefill_calls": 0,
            "decode_step_calls": 0,
        },
    }
    assert _selected_only(delta, "chemistry")
    delta["python"]["prefill_calls"] = 1
    assert not _selected_only(delta, "chemistry")
    delta["python"]["prefill_calls"] = 0
    delta["chemistry"]["prefill_calls"] = 0
    assert not _selected_only(delta, "chemistry")
