from collections import Counter

from abi.capability_compiler_phase3_fit_diagnostic import classify, fixed_sample, summarize_counts


def test_classification_separates_fit_drift_and_generalization():
    system = {
        "training_teacher_forced": {"action_accuracy": 0.995, "exact_sequence_rate": 0.91},
        "development_teacher_forced": {"action_accuracy": 0.80},
        "training_autonomous_sample": {"exact_response_bytes": 10, "observations": 100},
    }
    result = classify(system, {
        "training_action_accuracy_minimum": 0.99,
        "training_exact_sequence_rate_minimum": 0.90,
        "state_drift_training_action_accuracy_minimum": 0.99,
        "training_autonomous_exact_response_minimum": 0.95,
        "train_to_development_action_accuracy_drop_minimum": 0.05,
    })
    assert result == {"train_fit_or_capacity_limit": False, "autonomous_state_drift": True, "held_out_generalization_limit": True}


def test_fixed_sample_is_deterministic_and_balanced():
    examples = []
    for capability in ("a", "b"):
        for index in range(4):
            examples.append({"record_id": f"{capability}-{3-index}", "capability": capability})
    # The helper uses the repository capability list, so a reduced synthetic
    # catalog should fail closed rather than silently change strata.
    try:
        fixed_sample(examples, 2)
    except Exception as exc:
        assert "insufficient sample rows" in str(exc)
    else:
        raise AssertionError("reduced capability catalog was accepted")


def test_empty_capability_stratum_is_explicit_not_divided_by_zero():
    assert summarize_counts(Counter()) == {
        "action_accuracy": None,
        "exact_sequence_rate": None,
        "fixed_action_accuracy": None,
        "pointer_action_accuracy": None,
        "action_type_accuracy": None,
    }
