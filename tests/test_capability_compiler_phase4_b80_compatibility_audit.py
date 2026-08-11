from pathlib import Path

from abi.capability_compiler_phase4_b80_compatibility_audit import audit


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_B80_COMPATIBILITY_AUDIT_PROTOCOL_V624.json"


def test_complete_matrix_attributes_coadaptation() -> None:
    result = audit(ROOT, PROTOCOL)
    assert result["status"] == "PASS_ATTRIBUTION_STRONG_PARENT_BRIDGE_COADAPTATION"
    assert result["off_diagonal_passes"] == 0
    assert result["diagonal_advantage"] > result["bridge_main_effect_range"] > result["parent_main_effect_range"]
