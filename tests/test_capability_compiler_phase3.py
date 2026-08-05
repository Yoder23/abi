import copy
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import (
    CAPABILITIES,
    CAPABILITY_TO_ROUTE,
    PHASE1_IR_SHA256,
    Phase3Error,
    _deranged_outputs,
    _protocol,
    _route,
    _BalancedSampler,
    load_phase1_ir,
)


ROOT = Path(__file__).resolve().parents[1]
IR = ROOT / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir"


def _row(capability: str, index: int) -> dict:
    return {
        "capability": capability,
        "ir_record_id": f"{capability}-{index}",
        "normalized_output": f"output-{capability}-{index}",
        "normalized_generation_prompt_sha256": f"{index:064x}",
    }


def test_certified_phase1_ir_is_the_only_training_inventory():
    assert PHASE1_IR_SHA256 == "a246a52bcf27609b46cdb0530f1daaefe749b7c4a1000f9578f20e505a596f20"
    rows = load_phase1_ir(IR)
    assert len(rows) == 7_000
    assert {row["capability"] for row in rows} == set(CAPABILITIES)
    assert all(row["destination"] == "english_core" for row in rows)
    assert all(row["domain_labels"] == [] for row in rows)


def test_a0_routes_are_label_causal_and_a4_is_monolithic():
    for capability in CAPABILITIES:
        row = _row(capability, 1)
        assert _route("A0", row) == CAPABILITY_TO_ROUTE[capability]
        assert _route("A2", row) == CAPABILITY_TO_ROUTE[capability]
        assert _route("A3", row) == CAPABILITY_TO_ROUTE[capability]
        assert _route("A4", row) == 0
    assert set(CAPABILITY_TO_ROUTE.values()) == set(range(6))


def test_a1_route_is_deterministic_and_label_free():
    left = _row("grammar", 9)
    right = _row("rewriting", 9)
    assert _route("A1", left) == _route("A1", right)
    assert 0 <= _route("A1", left) < 6


def test_a2_derangement_changes_every_target_within_capability():
    rows = [_row(capability, index) for capability in CAPABILITIES for index in range(4)]
    mapping = _deranged_outputs(rows, 104729)
    expected_by_capability = {
        capability: {row["normalized_output"] for row in rows if row["capability"] == capability}
        for capability in CAPABILITIES
    }
    for row in rows:
        assert mapping[row["ir_record_id"]] != row["normalized_output"]
        assert mapping[row["ir_record_id"]] in expected_by_capability[row["capability"]]


def test_protocol_fails_closed_on_final_access_and_changed_binding(tmp_path):
    bound = tmp_path / "bound.txt"
    bound.write_text("bound", encoding="utf-8")
    import hashlib

    protocol = {
        "format": "abi-capability-compiler-phase3-protocol/1",
        "status": "PREREGISTERED_CONDITIONAL_PHASE3",
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED",
        "final_test_access": "PROHIBITED",
        "bindings": {"bound.txt": hashlib.sha256(b"bound").hexdigest()},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    loaded, _ = _protocol(tmp_path, path)
    assert loaded["final_test_access"] == "PROHIBITED"

    changed = copy.deepcopy(protocol)
    changed["final_test_access"] = "ALLOWED"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(Phase3Error, match="final-test firewall"):
        _protocol(tmp_path, path)

    path.write_text(json.dumps(protocol), encoding="utf-8")
    bound.write_text("changed", encoding="utf-8")
    with pytest.raises(Phase3Error, match="binding changed"):
        _protocol(tmp_path, path)


def test_repair_overlay_allows_only_the_measured_context_change(tmp_path):
    import hashlib

    bound = tmp_path / "bound.txt"
    bound.write_text("bound", encoding="utf-8")
    parent = {
        "format": "abi-capability-compiler-phase3-protocol/1",
        "status": "PREREGISTERED_CONDITIONAL_PHASE3",
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED",
        "final_test_access": "PROHIBITED",
        "training": {"max_tokens": 256},
        "bindings": {"bound.txt": hashlib.sha256(b"bound").hexdigest()},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    repair = {
        "format": "abi-capability-compiler-phase3-protocol-repair/1",
        "status": "PREREGISTERED_SINGLE_ALLOWED_REPAIR",
        "parent_protocol": {"path": "parent.json", "sha256": parent_sha},
        "changes": {"training.max_tokens": {"from": 256, "to": 512}},
        "bindings": {},
    }
    repair_path = tmp_path / "repair.json"
    repair_path.write_text(json.dumps(repair), encoding="utf-8")
    loaded, _ = _protocol(tmp_path, repair_path)
    assert loaded["training"]["max_tokens"] == 512
    assert loaded["repair"]["single_allowed_repair_consumed"] is True

    repair["changes"]["training.steps"] = {"from": 7000, "to": 14000}
    repair_path.write_text(json.dumps(repair), encoding="utf-8")
    with pytest.raises(Phase3Error, match="expanded beyond"):
        _protocol(tmp_path, repair_path)


def test_emitter_amendment_cannot_change_training_semantics(tmp_path):
    import hashlib

    bound = tmp_path / "bound.txt"
    bound.write_text("bound", encoding="utf-8")
    parent = {
        "format": "abi-capability-compiler-phase3-protocol/1",
        "status": "PREREGISTERED_CONDITIONAL_PHASE3",
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED",
        "final_test_access": "PROHIBITED",
        "bindings": {"bound.txt": hashlib.sha256(b"bound").hexdigest()},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    amendment = {
        "format": "abi-capability-compiler-phase3-evidence-emitter-amendment/1",
        "status": "PREREGISTERED_EVIDENCE_EMITTER_ONLY",
        "parent_protocol": {
            "path": "parent.json",
            "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        },
        "changes": {
            "A3.post_training_guard": {
                "from": "require_task_cake_byte_identity",
                "to": "accept_and_report_registered_scope_adamw_weight_decay",
            }
        },
        "bindings": {},
    }
    path = tmp_path / "amendment.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    loaded, _ = _protocol(tmp_path, path)
    assert loaded["evidence_emitter_amendment"]["training_semantics_changed"] is False

    amendment["changes"]["training.learning_rate"] = {"from": 0.0001, "to": 0.001}
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(Phase3Error, match="experiment semantics"):
        _protocol(tmp_path, path)


def test_balanced_sampler_is_identical_across_system_views():
    rows = []
    for capability in CAPABILITIES:
        for index in range(8):
            rows.append({"capability": capability, "record_id": f"{capability}-{index}"})
    left = _BalancedSampler(rows, 104729)
    right = _BalancedSampler(rows, 104729)
    for _ in range(100):
        assert [row["record_id"] for row in left.batch(4)] == [
            row["record_id"] for row in right.batch(4)
        ]


def test_paired_sampler_amendment_rejects_budget_changes(tmp_path):
    import hashlib

    bound = tmp_path / "bound.txt"
    bound.write_text("bound", encoding="utf-8")
    parent = {
        "format": "abi-capability-compiler-phase3-protocol/1",
        "status": "PREREGISTERED_CONDITIONAL_PHASE3",
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED",
        "final_test_access": "PROHIBITED",
        "bindings": {"bound.txt": hashlib.sha256(b"bound").hexdigest()},
    }
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    amendment = {
        "format": "abi-capability-compiler-phase3-paired-sampler-amendment/1",
        "status": "PREREGISTERED_PAIRED_CONFORMANCE_CORRECTION",
        "parent_protocol": {
            "path": "parent.json",
            "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        },
        "changes": {
            "successful_step_sampling": {
                "from": "sampler_advances_on_every_attempt",
                "to": "retry_identical_batch_until_optimizer_step_succeeds",
            },
            "successful_record_sequence_sha256": {
                "from": "absent",
                "to": "required_and_equal_across_A0_A1_A2_A3_A4",
            },
        },
        "bindings": {},
    }
    path = tmp_path / "paired.json"
    path.write_text(json.dumps(amendment), encoding="utf-8")
    loaded, _ = _protocol(tmp_path, path)
    assert loaded["paired_sampler_amendment"]["successful_record_sequence_equality_required"]
    amendment["changes"]["training.steps"] = {"from": 7000, "to": 8000}
    path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(Phase3Error, match="expanded"):
        _protocol(tmp_path, path)
