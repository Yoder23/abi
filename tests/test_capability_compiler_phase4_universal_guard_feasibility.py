from abi.capability_compiler_phase3_contract_guard_v2_audit import truncate_at_first_v2_collapse


def test_frozen_guard_truncates_repetition_without_rewriting_prefix():
    value, terminated = truncate_at_first_v2_collapse("To ensure Priya is ready Priya is ready for Priya is ready for Priya is ready for Priya is ready for goodness.")
    assert terminated
    assert value.startswith("To ensure Priya")
