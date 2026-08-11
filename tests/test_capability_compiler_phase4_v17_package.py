import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_v17_package import preflight


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_V17_PACKAGE_PROTOCOL_V636.json"


def test_historical_v17_protocol_fails_closed_after_layercake_evolves():
    with pytest.raises(Phase3Error, match="binding changed"):
        preflight(ROOT, PROTOCOL)


def test_changed_component_binding_fails_closed(tmp_path):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["bindings"][protocol["components"]["residual"]] = "0" * 64
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="binding changed"):
        preflight(ROOT, changed)
