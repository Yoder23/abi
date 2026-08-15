import hashlib

from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase2_common import CAPABILITIES
from abi.capability_compiler_phase4_b50_v24_l1_conformed_compose import (
    FORMAT,
    SEEDS,
    _evidence_valid,
    _paired_quality,
)


def test_composition_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-v24-l1-conformed-compose/1"
    assert SEEDS == (104729, 130363, 155921)


def test_evidence_digest_rejects_mutation():
    value = {"status": "PASS"}
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    assert _evidence_valid(value)
    assert not _evidence_valid({**value, "status": "FAIL"})


def test_paired_quality_direction():
    candidate = [
        {
            "probe_id": f"{capability}-{index}",
            "capability": capability,
            "functional_pass_v1": True,
        }
        for capability in CAPABILITIES
        for index in range(100)
    ]
    baseline = [
        {
            "probe_id": f"{capability}-{index}",
            "capability": capability,
            "functional_pass_v1": index < 90,
        }
        for capability in CAPABILITIES
        for index in range(100)
    ]
    result = _paired_quality(candidate, baseline, replicates=200, seed=77)
    assert result["lower_95"] > 0
    assert result["observations"] == 1400
