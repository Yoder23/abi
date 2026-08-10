from pathlib import Path


def test_fixed_budget_hybrid_audit_preserves_linear_and_nonlinear_paths():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_fixed_budget_hybrid_coefficient_audit.py").read_text(encoding="utf-8")
    assert "torch.linalg.svd(full_linear" in text
    assert "coordinate @ linear_output + F.silu(coordinate) @ nonlinear_output" in text
    assert '"artifact_written": False' in text
    assert '"final_test_accessed": False' in text
