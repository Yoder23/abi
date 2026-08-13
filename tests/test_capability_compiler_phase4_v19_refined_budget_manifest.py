from pathlib import Path

from abi.capability_compiler_phase4_v19_refined_budget_manifest import run


ROOT = Path(__file__).resolve().parents[1]


def test_refined_manifest_is_nested_and_preserves_bracket():
    result = run(ROOT, ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_V19_REFINED_BUDGET_MANIFEST_PROTOCOL_V740.json")
    assert result["status"] == "PASS_REFINED_B40_B80_NESTED_BUDGET_MANIFEST"
    assert [row["id"] for row in result["budgets"]] == ["B40", "B50", "B60", "B70", "B80"]
    assert result["gates"]["b40_identity_preserved"]
    assert result["gates"]["b80_identity_preserved"]
