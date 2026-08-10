from abi.capability_compiler_phase3_mlp_aware_attention_fit import FORMAT
def test_mlp_aware_fit_format_is_local_and_versioned() -> None: assert "attention" in FORMAT and FORMAT.endswith("/1")
