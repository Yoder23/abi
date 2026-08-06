from abi.capability_compiler_phase3_auxiliary_router_diagnostic import _summary


def test_summary_preserves_prediction_mode():
    rows = [{"correct": True, "predicted": "grammar", "capability": "grammar"}, {"correct": False, "predicted": "grammar", "capability": "coherence"}]
    result = _summary(rows)
    assert result["accuracy"] == 0.5
    assert result["predicted_counts"] == {"grammar": 2}
