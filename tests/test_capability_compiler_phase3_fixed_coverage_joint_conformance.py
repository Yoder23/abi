from pathlib import Path


def test_fixed_coverage_joint_conformance_preserves_validation():
    text=(Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_fixed_coverage_joint_conformance.py").read_text(encoding="utf-8")
    assert "ranked[:30]+ranked[32:302]" in text
    assert "ranked[30:32]" in text
    assert "joint.execute" in text
