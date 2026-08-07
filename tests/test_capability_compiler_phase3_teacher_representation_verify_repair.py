import hashlib

import pytest

from abi.capability_compiler_phase2_common import canonical_json_bytes
from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_teacher_representation_verify_repair import _verify_numerics_result


def _evidence() -> dict:
    value = {
        "status": "DIAGNOSTIC_COMPLETE_NONPROMOTIONAL",
        "artifact_verified": False,
        "training_performed": False,
        "original_batch_recomputation": {
            "vectors": 56,
            "maximum_absolute_error": 0.0,
            "mean_absolute_error": 0.0,
            "mean_exact_scalar_fraction": 1.0,
            "minimum_cosine_similarity": 0.999999,
        },
        "singleton_recomputation": {"maximum_absolute_error": 0.05},
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def test_repair_requires_exact_batch_reproduction_and_preserves_failure() -> None:
    expected = {"sample_vectors": 56, "minimum_cosine_similarity": 0.99999, "failed_singleton_threshold": 0.0078125}
    _verify_numerics_result(_evidence(), expected)
    damaged = _evidence()
    damaged["original_batch_recomputation"]["maximum_absolute_error"] = 0.001
    damaged["evidence_sha256"] = hashlib.sha256(canonical_json_bytes({k: v for k, v in damaged.items() if k != "evidence_sha256"})).hexdigest()
    with pytest.raises(Phase3Error):
        _verify_numerics_result(damaged, expected)
