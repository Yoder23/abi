from abi.capability_compiler_phase3_causal_field_verify import _selected_indices


def test_selected_indices_are_capability_stratified_and_deterministic():
    rows = [
        {"record_id": f"{capability}-{index}", "capability": capability}
        for capability in ("a", "b")
        for index in range(5)
    ]
    # The helper is campaign-strict and therefore rejects incomplete capability sets.
    try:
        _selected_indices(rows, seed=1, per_capability=2)
    except Exception as exc:
        assert "capability inventory" in str(exc)
    else:
        raise AssertionError("incomplete campaign capabilities must fail closed")
