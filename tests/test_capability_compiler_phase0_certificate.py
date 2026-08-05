import copy
import json
from pathlib import Path

from abi.capability_compiler_phase0_certificate import validate_certificate


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = json.loads(
    (ROOT / "ABI_CAPABILITY_COMPILER_PHASE0_CERTIFICATE_V1.json").read_text(encoding="utf-8")
)


def test_checked_in_phase0_certificate_verifies_immutable_commit():
    assert validate_certificate(CERTIFICATE, ROOT) == []


def test_certificate_rejects_mutated_hash():
    mutated = copy.deepcopy(CERTIFICATE)
    mutated["implementation_locks"]["ABI_CAPABILITY_COMPILER_PHASE0_PROTOCOL_V1.json"] = "0" * 64
    errors = validate_certificate(mutated, ROOT)
    assert any("implementation hash mismatch" in error for error in errors)


def test_certificate_rejects_training_and_premature_phase_transition():
    mutated = copy.deepcopy(CERTIFICATE)
    mutated["new_training_performed"] = True
    mutated["phase_transition"]["phase2_through_phase8"] = "OPEN"
    errors = validate_certificate(mutated, ROOT)
    assert "Phase 0 cannot include new training" in errors
    assert "later phases must remain locked" in errors


def test_certificate_rejects_failed_certification_suite():
    mutated = copy.deepcopy(CERTIFICATE)
    mutated["verification"]["certification_tree_full_test_suite"]["failed"] = 1
    errors = validate_certificate(mutated, ROOT)
    assert "certification tree full suite did not pass" in errors


def test_current_documents_record_phase3_branch_failure_and_phase4_locked():
    status = (ROOT / "CURRENT_PROJECT_STATUS.md").read_text(encoding="utf-8")
    mission = (ROOT / "ACTIVE_MISSION.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    for document in (status, mission, roadmap):
        assert "Phase 1" in document
    assert "Phase 0 is **COMPLETE**" in status
    assert "Phase 1 is **COMPLETE**" in status
    assert "Phase 2" in status and "**BLOCKED_EXTERNAL_HUMAN_RATINGS**" in status
    assert "Phase 3" in status and "COMPLETE_FAILED" in status
    assert "Phase 4" in status and "locked" in status.lower()
    assert "not certified" in status.lower()
