from abi.capability_compiler_phase3_routed_v15_layer2_source_aligned_repair import FORMAT


def test_routed_v15_layer2_source_aligned_format_is_versioned() -> None:
    assert "source-aligned-repair" in FORMAT
    assert FORMAT.endswith("/2")
