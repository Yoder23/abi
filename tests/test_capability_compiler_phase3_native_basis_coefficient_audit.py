from pathlib import Path


def test_native_basis_audit_uses_realizable_existing_maps():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_native_basis_coefficient_audit.py").read_text(encoding="utf-8")
    assert "linear_map" in text
    assert "route_maps" in text
    assert "artifact_written\": False" in text
