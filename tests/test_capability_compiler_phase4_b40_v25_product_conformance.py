from abi.capability_compiler_phase4_b40_v25_product_conformance import (
    SEEDS,
    _runtime_protocol,
)


def test_b40_seed_order_is_paired_and_fixed():
    assert SEEDS == (104729, 130363, 155921)


def test_runtime_protocol_changes_only_system_paths():
    protocol = {"router_config": "old", "guard_artifact": "old", "x": 1}
    runtime = _runtime_protocol(
        protocol, {"router_config": "router", "guard_artifact": "guard"}
    )
    assert runtime == {"router_config": "router", "guard_artifact": "guard", "x": 1}
    assert protocol["router_config"] == "old"
