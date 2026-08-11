import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_v18_package import preflight


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_V18_PACKAGE_PROTOCOL_V638.json"


def test_live_preflight_exact_component_and_residual_inventory():
    result = preflight(ROOT, PROTOCOL)
    assert result["status"] == "PASS_PREFLIGHT"
    assert result["component_parameters"] == {"model": 61655050, "router": 1058040, "residual": 99840}
    assert result["tensor_namespaces"] == {"model": 82, "router": 3, "residual": 4}
    assert result["residual_keys"] == ["down", "norm.bias", "norm.weight", "up"]
    assert not result["training_performed"]
    assert not result["final_test_accessed"]


def test_changed_component_binding_fails_closed(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["bindings"][protocol["components"]["residual"]] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="binding changed"):
        preflight(ROOT, changed)
