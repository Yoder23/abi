import hashlib

from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase4_b50_v24_runtime_compose import (
    FORMAT,
    RESULT_FORMAT,
    _wrapper_digest_valid,
)


def test_v24_compose_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-v24-runtime-compose/1"
    assert RESULT_FORMAT == (
        "abi-capability-compiler-phase4-b50-v24-runtime-compose-result/1"
    )


def test_v24_wrapper_digest_excludes_only_digest_field():
    value = {"format": "x", "status": "PASS"}
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    assert _wrapper_digest_valid(value)
    assert not _wrapper_digest_valid({**value, "status": "FAIL"})
