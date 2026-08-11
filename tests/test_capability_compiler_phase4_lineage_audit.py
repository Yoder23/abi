import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_lineage_audit import load_protocol, run


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_LINEAGE_AUDIT_PROTOCOL_V557.json"


def test_real_lineage_audit_passes_and_detects_hidden_warm_start() -> None:
    result = run(ROOT, PROTOCOL)
    assert result["status"] == "PASS_COMPLETE_LINEAGE_AUDIT_FRONTIER_PROTOCOL_REQUIRED"
    assert result["confounds"]["varying_only_v526_v480_subset_is_invalid"] is True
    assert result["gates"]["final_test_accessed"] is False
    assert result["unique_imported_information"]["source_attempts"] > 14000


def test_tampered_binding_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    first = next(iter(protocol["bindings"]))
    protocol["bindings"][first] = "0" * 64
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="binding changed"):
        load_protocol(ROOT, path)


def test_training_authorization_mutation_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["neural_training_authorized"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(ROOT, path)
