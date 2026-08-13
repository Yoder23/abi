import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_v19_package import load_protocol, preflight


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_V19_PACKAGE_PROTOCOL_V711.json"


def test_live_preflight_preserves_v18_payload_and_adds_v19_host_contract():
    result = preflight(ROOT, PROTOCOL)
    assert result["status"] == "PASS_V19_REPACKAGING_PREFLIGHT"
    assert result["source_payload_hash"] == "d19e254846544166dfaf814642d51fa2f1ba595404e8ac02fa43e69555440989"
    assert result["tensor_count"] == 89
    assert result["total_parameters"] == 62812930
    assert all(result["gates"].values())
    assert not result["training_performed"]
    assert not result["teacher_present"]


def test_changed_source_binding_fails_closed(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["bindings"][protocol["source_package"]] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="binding changed"):
        load_protocol(ROOT, changed)
