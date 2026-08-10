from pathlib import Path


def test_recurring_outlier_attribution_is_read_only_and_checks_context_coverage():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_recurring_outlier_attribution.py").read_text(encoding="utf-8")
    assert "outside_capability_train_full_length_support" in text
    assert "source_host_mapping_valid" in text
    assert '"training_performed": False' in text
    assert '"artifact_written": False' in text
