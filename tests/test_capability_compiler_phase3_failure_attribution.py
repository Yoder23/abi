from abi.capability_compiler_phase3_failure_attribution import classify_attribution


RULES = {
    "minimum_nll_reduction_vs_no_payload": 0.10,
    "minimum_accuracy_gain_vs_no_payload": 0.05,
    "minimum_corrupted_nll_ratio": 1.10,
    "minimum_corrupted_accuracy_drop": 0.05,
    "native_new_suite_functional_floor": 0.90,
}


def _metrics():
    return {
        "external_layercake_control": {"identity_pass": True, "sealed_certificates_pass": True},
        "systems": {
            "C0": {"teacher_forced": {"mean_nll": 1.0, "token_accuracy": 0.70}},
            "C3": {"teacher_forced": {"mean_nll": 2.0, "token_accuracy": 0.20}},
            "P0": {"autonomous": {"functional_rate": 0.95, "repetition_collapses": 0}},
        },
        "c0_corruption_recovery": {"aggregate": {"nll_ratio_corrupted_to_clean": 1.20, "accuracy_delta_corrupted_minus_clean": -0.10}},
    }


def test_identity_failure_stops_abi_and_assigns_layercake():
    value = _metrics()
    value["external_layercake_control"]["identity_pass"] = False
    result = classify_attribution(value, RULES)
    assert result["primary"] == "LAYERCAKE_HOST_REGRESSION_OR_IDENTITY_FAILURE"
    assert result["owners"] == ["LAYERCAKE"]


def test_signal_plus_prefix_sensitivity_assigns_integration_repair():
    result = classify_attribution(_metrics(), RULES)
    assert result["primary"] == "ABI_SIGNAL_PRESENT_INTEGRATION_STATE_RECOVERY_LIMITING"
    assert result["abi_teacher_payload_signal"] is True
    assert result["host_representational_ceiling_proven"] is False


def test_missing_payload_signal_is_abi_deficit_not_layercake_failure():
    value = _metrics()
    value["systems"]["C0"]["teacher_forced"] = {"mean_nll": 1.9, "token_accuracy": 0.22}
    result = classify_attribution(value, RULES)
    assert result["primary"] == "ABI_ACQUISITION_INFORMATION_DEFICIT"
    assert result["layercake_regression"] is False
    assert "ABI" in result["owners"]


def test_new_suite_gap_is_never_relabeled_as_regression():
    value = _metrics()
    value["systems"]["P0"]["autonomous"] = {"functional_rate": 0.50, "repetition_collapses": 3}
    result = classify_attribution(value, RULES)
    assert result["native_new_suite_scope_gap"] is True
    assert result["layercake_regression"] is False
    assert "LAYERCAKE_SCOPE_REVIEW" in result["owners"]
