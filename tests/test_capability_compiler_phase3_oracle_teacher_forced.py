from abi.capability_compiler_phase3_oracle_teacher_forced import classify


THRESHOLDS = {"teacher_forced_token_accuracy_minimum": 0.95, "teacher_forced_mean_nll_maximum": 0.2}


def test_high_teacher_forced_fit_assigns_autonomous_state():
    assert classify({"token_accuracy": 0.97, "mean_nll": 0.1}, THRESHOLDS) == "AUTONOMOUS_STATE_DYNAMICS_LIMITATION"


def test_low_teacher_forced_fit_does_not_claim_host_ceiling():
    assert classify({"token_accuracy": 0.90, "mean_nll": 0.3}, THRESHOLDS) == "BRIDGE_FIT_OPTIMIZATION_OR_EXPRESSIVITY_LIMITATION"
