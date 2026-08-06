from abi.capability_compiler_phase3_bpe_failure_attribution import classify, fact_free_mode


THRESHOLDS = {
    "training_autonomous_exact_minimum": 0.9,
    "header_intervention_improvement_minimum": 0.2,
}


def test_fact_free_mode_requires_both_markers():
    assert fact_free_mode("PAVO-X belongs to NORU-X")
    assert not fact_free_mode("PAVO-X")


def test_classification_prioritizes_training_state_failure():
    assert classify(training_exact_rate=0.8, same_header_delta=0.5, body_only_delta=0.5, thresholds=THRESHOLDS) == "PRIMARY_MODEL_FIT_OR_AUTONOMOUS_STATE"


def test_classification_detects_header_shift_after_training_replay_passes():
    assert classify(training_exact_rate=0.95, same_header_delta=0.3, body_only_delta=0.0, thresholds=THRESHOLDS) == "PRIMARY_ACQUISITION_TO_EVALUATION_HEADER_COVARIATE_SHIFT"
