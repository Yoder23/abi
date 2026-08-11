from abi.capability_compiler_phase3_route_isolated_host_reproduction import assess_hosts


def _host():
    return {"checkpoint_sha256": "c", "raw_outputs_sha256": "o", "functional_passes_v1": 1393, "repetition_collapses_v2": 0, "router_correct": 1400, "strong_routes_exact": 1000, "observations": 1400, "final_test_accessed": False}


def test_assess_hosts_requires_two_exact_hosts():
    assert all(assess_hosts("o", "c", [_host(), _host()]).values())
    assert not assess_hosts("o", "c", [_host()])["two_fresh_hosts_present"]


def test_assess_hosts_rejects_semantic_mutation():
    changed = _host()
    changed["raw_outputs_sha256"] = "x"
    assert not assess_hosts("o", "c", [_host(), changed])["byte_identical_outputs_to_reference"]
