from abi.capability_compiler_phase3_routed_v15_progressive_extract import FORMAT


def test_routed_v15_progressive_format_is_versioned() -> None:
    assert "routed-v15-progressive" in FORMAT
    assert FORMAT.endswith("/1")
