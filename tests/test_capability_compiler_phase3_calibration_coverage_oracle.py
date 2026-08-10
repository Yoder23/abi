from pathlib import Path


def test_coverage_oracle_preserves_validation_and_uses_one_train_scale():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_calibration_coverage_oracle.py").read_text(encoding="utf-8")
    assert "ranked[:30] + ranked[32:302]" in text
    assert "ranked[30:32]" in text
    assert "oracle.execute" in text
