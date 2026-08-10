from abi.capability_compiler_phase3_routed_v15_layer1_extract import FORMAT


def test_routed_v15_layer1_format_is_versioned() -> None:
    assert "routed-v15-layer1" in FORMAT
    assert FORMAT.endswith("/1")
