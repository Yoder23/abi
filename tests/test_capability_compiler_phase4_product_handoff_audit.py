import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_product_handoff_audit import audit


ROOT = Path(__file__).resolve().parents[1]
LAYERCAKE = ROOT.parent / "layercake_release"
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_PRODUCT_HANDOFF_AUDIT_PROTOCOL_V632.json"


def test_live_handoff_audit_fails_closed_on_incompatible_composite():
    result = audit(ROOT, LAYERCAKE, PROTOCOL)
    assert result["status"] == "FAIL_NO_DECLARED_LAYERCAKE_HANDOFF_ACCEPTS_PHASE3_COMPOSITE"
    assert result["checks"]["phase3_machine_evidence_complete"]
    assert result["checks"]["layercake_product_identity_consistent"]
    assert result["checks"]["legacy_and_product_checkpoints_distinct"]
    assert not result["checks"]["phase3_endpoint_is_single_signed_package"]
    assert not result["checks"]["some_declared_handoff_accepts_phase3_endpoint_unchanged"]
    assert not result["phase4_certified"]


def test_all_declared_interfaces_are_rejected():
    result = audit(ROOT, LAYERCAKE, PROTOCOL)
    assert set(result["declared_handoffs"]) == {
        "lc-direct-neural-core/2",
        "lc-direct-neural-core/5",
        "lc-direct-neural-core/16",
    }
    assert not any(value["compatible"] for value in result["declared_handoffs"].values())


def test_protocol_rejects_changed_binding(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["abi_bindings"]["ABI_CAPABILITY_COMPILER_PHASE3_FINAL_CERTIFICATE_AUDIT_RESULT_V551.json"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="ABI binding changed"):
        audit(ROOT, LAYERCAKE, changed)
