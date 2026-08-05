import copy
import hashlib

import pytest

from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase3_sequence_verifier import (
    SequenceVerificationError,
    _verify_decision,
    _verify_metadata,
)


def _rehash(value, field):
    body = copy.deepcopy(value)
    body.pop(field, None)
    value[field] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def test_metadata_verifier_rejects_rehashed_teacher_presence():
    metadata = {
        "system": "B0",
        "seed": 104729,
        "protocol_sha256": "p",
        "final_test_accessed": False,
        "phase2_human_gate": "DEFERRED_NOT_PASSED",
        "source": {
            "phase1_ir_sha256": "a246a52bcf27609b46cdb0530f1daaefe749b7c4a1000f9578f20e505a596f20",
            "teacher_present_during_training": True,
            "teacher_present_at_inference": False,
            "source_parameters_copied": 0,
            "source_blocks_retained": 0,
        },
    }
    _rehash(metadata, "manifest_sha256")
    with pytest.raises(SequenceVerificationError, match="teacher present"):
        _verify_metadata(metadata, system="B0", protocol_sha="p")


def test_decision_verifier_rejects_rehashed_false_promotion():
    decision = {
        "status": "FAIL_INITIAL_SEED_SEQUENCE_SUCCESSOR",
        "protocol": {"sha256": "p"},
        "phase3_certified": False,
        "phase4_status": "LOCKED",
        "final_test_accessed": False,
        "decision": {"branch_promoted": True, "remaining_two_seeds_authorized": False},
        "negative_evidence_preserved": True,
        "gates": {"quality": False},
    }
    _rehash(decision, "evidence_sha256")
    with pytest.raises(SequenceVerificationError, match="promoted"):
        _verify_decision(decision, protocol_sha="p")
