import json

import pytest

from abi.capability_compiler_phase3_component_diagnostic import (
    ComponentDiagnosticError,
    EXPECTED_MUTATIONS,
    VARIANTS,
    load_diagnostic_protocol,
)


def test_component_diagnostic_scope_is_frozen():
    assert VARIANTS == ("R1", "R2", "R3")
    assert EXPECTED_MUTATIONS["R1"] == tuple(
        f"task_cakes.{route}.up.weight" for route in range(6)
    )
    assert EXPECTED_MUTATIONS["R2"] == (
        "abi_sequence_bridge.route_embedding.weight",
    )
    assert len(EXPECTED_MUTATIONS["R3"]) == 7


def test_component_diagnostic_protocol_fails_closed(tmp_path):
    protocol = {
        "format": "abi-capability-compiler-phase3-component-diagnostic/1",
        "status": "PREREGISTERED_DIAGNOSTIC_ONLY",
        "phase3_promotion_eligible": False,
        "final_test_access": "PROHIBITED",
        "variants": {
            "R0": "sealed B1 checkpoint and existing unmodified evaluation",
            "R1": "same B1 checkpoint with all six output cakes bypassed in memory",
            "R2": "same B1 checkpoint with route embedding bypassed in memory",
            "R3": "same B1 checkpoint with output cakes and route embedding bypassed in memory",
        },
        "bindings": {},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    load_diagnostic_protocol(tmp_path, path)
    protocol["phase3_promotion_eligible"] = True
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ComponentDiagnosticError, match="governance changed"):
        load_diagnostic_protocol(tmp_path, path)
