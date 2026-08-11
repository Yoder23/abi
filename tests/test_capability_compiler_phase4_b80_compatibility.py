from pathlib import Path

from abi.capability_compiler_phase4_b80_compatibility import preflight


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_B80_COMPATIBILITY_PROTOCOL_V615.json"


def test_preflight_seals_complete_three_by_three_matrix() -> None:
    result = preflight(ROOT, PROTOCOL)
    assert result["status"] == "PASS_B80_COMPATIBILITY_PREFLIGHT"
    assert result["matrix_cells"] == 9
    assert result["existing_diagonal_cells"] == 3
    assert result["authorized_off_diagonal_cells"] == 6
    assert result["training_performed"] is False
