from abi.capability_compiler_phase3_feature_alignment_audit import FORMAT
def test_alignment_audit_format_is_versioned() -> None: assert "alignment" in FORMAT and FORMAT.endswith("/1")
