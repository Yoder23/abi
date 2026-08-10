from abi.capability_compiler_phase3_three_route_label_oracle_audit import FORMAT, _route


def test_label_oracle_format_is_versioned() -> None:
    assert "three-route-label-oracle" in FORMAT
    assert FORMAT.endswith("/1")


def test_route_maps_only_two_specialists() -> None:
    specialists = ("abstention", "conversation")
    assert _route("abstention", specialists) == "abstention"
    assert _route("conversation", specialists) == "conversation"
    assert _route("grammar", specialists) == "generic"
