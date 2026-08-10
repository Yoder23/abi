from abi.capability_compiler_phase3_routed_v15_layer2_geometry_audit import FORMAT


def test_routed_v15_layer2_geometry_format_is_versioned() -> None:
    assert "layer2-geometry" in FORMAT
    assert FORMAT.endswith("/1")
