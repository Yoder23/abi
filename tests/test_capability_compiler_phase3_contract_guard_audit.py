from abi.capability_compiler_phase3_contract_guard_audit import apply_contract_guard, truncate_before_fifth_repeated_gram


def test_repetition_guard_truncates_before_fifth_occurrence():
    value, changed = truncate_before_fifth_repeated_gram("Please act now now now now now later")
    assert changed
    assert value == "Please act now now now now"


def test_abstention_clause_is_added_only_when_marker_absent():
    markers = ("cannot determine", "not enough information")
    clause = "I cannot determine that from the information given."
    value, changes = apply_contract_guard("The source does not specify it.", "abstention", markers, clause)
    assert value.startswith(clause)
    assert changes["abstention_clause_prefixed"]
    existing, existing_changes = apply_contract_guard("I cannot determine it.", "abstention", markers, clause)
    assert existing == "I cannot determine it."
    assert not existing_changes["abstention_clause_prefixed"]
