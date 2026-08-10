from pathlib import Path


def test_fixed_budget_nonlinear_audit_is_fail_fast_and_no_artifact():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_fixed_budget_nonlinear_coefficient_audit.py").read_text(encoding="utf-8")
    assert "F.silu(first(" in text
    assert "route_maps" in text
    assert '"artifact_written": False' in text
    assert '"final_test_accessed": False' in text
