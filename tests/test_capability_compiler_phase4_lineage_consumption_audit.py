import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_lineage_consumption_audit import load_protocol, run


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_LINEAGE_CONSUMPTION_AUDIT_PROTOCOL_V560.json"


def test_consumed_information_excludes_unread_targeted_records() -> None:
    result = run(ROOT, PROTOCOL)
    assert result["status"] == "PASS_CONSUMED_INFORMATION_LINEAGE_FRONTIER_PROTOCOL_OPEN"
    assert result["consumption_rules"]["v474_targeted_weak"]["records"] == 2000
    assert result["unused_but_archived"]["v138_nonweak_records"] == 5000
    assert result["consumed_unique_information"]["source_attempts"] < result["historical_v558"]["container_unique_source_attempts"]


def test_changed_historical_evidence_flag_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["historical_v558_changed"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(ROOT, path)
