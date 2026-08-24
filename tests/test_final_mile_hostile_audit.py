import hashlib
import json
from pathlib import Path

from abi.final_mile import canonical_json_bytes
from abi.final_mile_hostile_audit import audit

ROOT = Path(__file__).resolve().parents[1]


def test_hostile_audit_rejects_false_claims_and_preserves_blockers(tmp_path):
    result = audit(ROOT, output=tmp_path / "hostile.json")
    assert result["status"] == "FAIL_BLOCKING_FINDINGS"
    assert result["attacks_rejected"] == result["attacks_total"] == 15
    assert result["unresolved_blocking_findings"] >= 8
    assert result["final_status"] == "HOST_INDEPENDENCE_FAILED"


def test_hostile_audit_evidence_hash_replays(tmp_path):
    output = tmp_path / "hostile.json"
    audit(ROOT, output=output)
    value = json.loads(output.read_text(encoding="utf-8"))
    declared = value.pop("evidence_sha256")
    assert hashlib.sha256(canonical_json_bytes(value)).hexdigest() == declared
