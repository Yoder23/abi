from abi.capability_compiler_phase4_b50_v23_runtime_compose import (
    FORMAT,
    RESULT_FORMAT,
    _wrapper_digest_valid,
)
from abi.capability_compiler_phase2_common import canonical_json_bytes
import hashlib


def test_v23_compose_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-v23-runtime-compose/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b50-v23-runtime-compose-result/1"


def test_v23_wrapper_digest_excludes_only_digest_field():
    result = {"format": "x", "status": "PASS"}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    assert _wrapper_digest_valid(result)
    assert not _wrapper_digest_valid({**result, "status": "FAIL"})
