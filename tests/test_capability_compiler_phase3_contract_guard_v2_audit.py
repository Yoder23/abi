from abi.capability_compiler_phase3_contract_guard_v2_audit import truncate_at_first_v2_collapse
from abi.capability_compiler_repetition_v2 import repetition_collapse_v2


def test_exact_v2_guard_stops_before_four_repeat_loop():
    source = "Please act now now now now now later."
    assert repetition_collapse_v2(source)
    value, changed = truncate_at_first_v2_collapse(source)
    assert changed
    assert not repetition_collapse_v2(value)
    assert value == "Please act now now now"


def test_exact_v2_guard_leaves_noncollapsed_text_byte_exact():
    source = "Please send the green notebook by Friday."
    value, changed = truncate_at_first_v2_collapse(source)
    assert not changed
    assert value == source
