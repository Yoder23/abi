import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_uniform_matrix_audit import audit, load_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_UNIFORM_MATRIX_AUDIT_PROTOCOL_V604.json"


def test_complete_matrix_rejects_stabilization() -> None:
    result = audit(ROOT, PROTOCOL)
    assert result["status"] == "FAIL_STABILIZATION_REJECTED_NO_STABLE_FRONTIER"
    assert result["matrix"] == {"B40": ["FAIL", "PASS", "FAIL"], "B80": ["PASS", "FAIL", "PASS"]}
    assert result["matrix_changed_from_historical"] is False
    assert result["all_exposure_ranges_at_most_one"] is True


def test_final_access_mutation_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["final_test_access"] = "ALLOWED"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(ROOT, path)
