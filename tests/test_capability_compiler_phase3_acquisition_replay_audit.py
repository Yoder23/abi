from abi.capability_compiler_phase3_acquisition_replay_audit import literal_recall, selected_records


def test_literal_recall_uses_required_not_alternative_literals():
    evaluator = {
        "kind": "all_of",
        "rules": [
            {"kind": "contains_all", "values": ["N123", "Friday"]},
            {"kind": "contains_any", "values": ["please", "thank"]},
        ],
    }
    assert literal_recall("Please update N123.", evaluator) == 0.5


def test_selected_records_is_balanced_and_deterministic():
    capabilities = ("abstention", "coherence", "fluent_realization", "tone_control")
    rows = [
        {"record_id": f"{index:03d}-{capability}-{builder}", "capability": capability, "builder": builder}
        for capability in capabilities
        for builder in range(4)
        for index in range(5)
    ]
    selected = selected_records(rows, 2)
    assert len(selected) == 32
    assert {row["record_id"].split("-", 1)[0] for row in selected} == {"000", "001"}
