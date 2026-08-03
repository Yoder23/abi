from __future__ import annotations

import pytest

from abi.capability_naive_training_base import (
    CapabilityNaiveTrainingBaseError,
    build_training_metadata,
)


def _receiver() -> dict:
    tensors = [
        {
            "name": f"task_cakes.{route}.down.weight",
            "parameters": 10,
        }
        for route in range(10)
    ]
    tensors.append({"name": "transformer.weight", "parameters": 100})
    return {
        "format": "abi-layercake-capability-naive-receiver/1",
        "status": "SEALED_CAUSAL_NEGATIVE_CONTROL",
        "role": "capability_naive_receiver",
        "seed": 7,
        "manifest_sha256": "manifest",
        "layercake_host": {
            "architecture": {"task_cakes": 10},
            "canonical_semantic_abi_sha256": "abi",
        },
        "checkpoint": {
            "sha256": "checkpoint",
            "bytes": 42,
            "parameter_count": 200,
            "tensors": tensors,
        },
        "imported_information": {
            "foreign_teacher_parameters_copied": 0,
            "layercake_learned_parameters_copied": 0,
            "bridge_parameters": 0,
            "training_steps": 0,
            "training_tokens": 0,
        },
    }


def test_training_metadata_keeps_naive_state_and_counts_sparse_activity() -> None:
    metadata = build_training_metadata(
        _receiver(), receiver_manifest_sha256="file"
    )
    assert metadata["checkpoint"]["sha256"] == "checkpoint"
    assert metadata["parameters"] == {"total": 200, "active": 110}
    assert metadata["imported_information"]["training_tokens"] == 0


def test_training_metadata_rejects_a_learned_receiver() -> None:
    receiver = _receiver()
    receiver["imported_information"]["training_tokens"] = 1
    with pytest.raises(CapabilityNaiveTrainingBaseError):
        build_training_metadata(receiver, receiver_manifest_sha256="file")
