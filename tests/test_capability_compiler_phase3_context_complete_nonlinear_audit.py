from pathlib import Path


def test_context_complete_control_changes_only_calibration_cap():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_context_complete_nonlinear_audit.py").read_text(encoding="utf-8")
    assert "maximum != 512" in text
    assert "int(maximum_tokens) != 128" in text
    assert "nonlinear.execute" in text
