from abi.capability_compiler_phase3_routed_v15_layer0_extract import FORMAT, ROUTES, _route


def test_routed_v15_layer0_format_is_versioned() -> None:
    assert "routed-v15-layer0" in FORMAT
    assert FORMAT.endswith("/1")


def test_routed_v15_layer0_routes_are_exact() -> None:
    assert ROUTES == ("generic", "abstention", "conversation")
    assert _route("abstention") == 1
    assert _route("conversation") == 2
    assert _route("grammar") == 0
