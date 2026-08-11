from abi.capability_compiler_phase3_route_isolated_cpu_runtime import adapt_screen_protocol


def test_adapter_changes_only_candidate_checkpoint():
    screen = {"parent": {"x": 1}, "router": {"x": 2}, "guard": {"x": 3}, "candidate": {"checkpoint": "old", "checkpoint_sha256": "old"}}
    protocol = {"candidate": {"checkpoint": "new", "checkpoint_sha256": "hash"}}
    adapted = adapt_screen_protocol(screen, protocol)
    assert adapted["candidate"] == {"checkpoint": "new", "checkpoint_sha256": "hash", "immutable": True}
    assert adapted["parent"] == screen["parent"]
    assert screen["candidate"]["checkpoint"] == "old"
