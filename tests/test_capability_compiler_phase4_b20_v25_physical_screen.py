from abi.capability_compiler_phase4_b20_v25_physical_screen import (
    _route_for_capability,
    preservation_gates,
)


def test_v25_residual_mapping_is_fixed_and_sparse():
    assert _route_for_capability("abstention") == 0
    assert _route_for_capability("coherence") == 1
    assert _route_for_capability("fluent_realization") == 2
    assert _route_for_capability("tone_control") == 3
    assert _route_for_capability("clarification") == 4
    assert _route_for_capability("rewriting") == -1


def test_preservation_rejects_immutable_output_change():
    historical = {
        "a": {
            "probe_id": "a",
            "capability": "fluent_realization",
            "output": "exact",
            "functional_pass_v1": True,
            "repetition_collapse_v2": False,
        },
        "b": {
            "probe_id": "b",
            "capability": "clarification",
            "output": "old",
            "functional_pass_v1": False,
            "repetition_collapse_v2": False,
        },
    }
    rows = [
        {"probe_id": "a", "capability": "fluent_realization", "output": "changed", "functional_pass_v1": True},
        {"probe_id": "b", "capability": "clarification", "output": "new", "functional_pass_v1": True},
    ]
    gates = preservation_gates(historical, rows)
    assert not gates["changes_bounded_to_declared_host_scope"]
    assert not gates["all_immutable_outputs_exact"]
