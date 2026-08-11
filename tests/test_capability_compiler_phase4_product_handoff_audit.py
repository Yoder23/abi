import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_product_handoff_audit import audit


ROOT = Path(__file__).resolve().parents[1]
LAYERCAKE = ROOT.parent / "layercake_release"
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_PRODUCT_HANDOFF_AUDIT_PROTOCOL_V632.json"


def test_historical_audit_fails_closed_after_external_handoff_evolves():
    with pytest.raises(Phase3Error, match="binding changed"):
        audit(ROOT, LAYERCAKE, PROTOCOL)


def test_historical_protocol_retains_pre_v17_interface_set():
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert set(protocol["interfaces"]) == {"v2", "v5", "v16"}
    assert "v17" not in protocol["interfaces"]


def test_protocol_rejects_changed_binding(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["abi_bindings"]["ABI_CAPABILITY_COMPILER_PHASE3_FINAL_CERTIFICATE_AUDIT_RESULT_V551.json"] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="ABI binding changed"):
        audit(ROOT, LAYERCAKE, changed)
