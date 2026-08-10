from abi.capability_compiler_phase3_complete_mlp_interface_audit import FORMAT
def test_complete_mlp_audit_is_explicitly_diagnostic() -> None: assert "audit" in FORMAT and FORMAT.endswith("/1")
