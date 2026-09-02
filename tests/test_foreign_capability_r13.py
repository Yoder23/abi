from __future__ import annotations

import hashlib
import json

import pytest

from experiments.foreign_capability_r13.core import (
    R13Error,
    capability_rows,
    evidence_hash,
)
from experiments.foreign_capability_r13.recipient_worker import _summarize
from experiments.foreign_capability_r13.verify import _evidence, _probabilities
from experiments.native_transfer_r8.capability_generator import (
    committed_heldout_capabilities,
)


def _config() -> dict:
    return {
        "data": {
            "training_rows_per_capability": 2904,
            "training_depths": [1, 2, 3, 4, 5],
            "training_seed": 13013002,
            "evaluation_rows_per_capability": 32,
            "evaluation_depths": [6, 7],
            "evaluation_seed": 13013003,
        }
    }


def test_r13_rows_are_unique_disjoint_and_complete_atomic() -> None:
    secret = bytes(range(32))
    commitment = hashlib.sha256(secret).hexdigest()
    capabilities = committed_heldout_capabilities(
        secret.hex(), expected_commitment=commitment, count=2
    )
    training, evaluation, atomic = capability_rows(_config(), capabilities)
    for train, test, atoms in zip(training, evaluation, atomic):
        assert len(train) == 2904
        assert len(test) == 32
        assert len(atoms) == 24
        assert {row["prompt_sha256"] for row in train}.isdisjoint(
            {row["prompt_sha256"] for row in test}
        )


def test_r13_evidence_fails_closed() -> None:
    receipt = {"format": "example", "count": 1}
    receipt["evidence_sha256"] = evidence_hash(receipt)
    _evidence(receipt, "example")
    receipt["count"] = 2
    with pytest.raises(R13Error, match="evidence hash changed"):
        _evidence(receipt, "example")


def test_r13_recipient_summary_detects_removal_difference() -> None:
    evaluation = [[{"row_id": "row", "answer": 1}]]
    rows = [
        {
            "capability_id": "cap",
            "condition": condition,
            "row_id": "row",
            "canonical_prediction": 1,
            "prediction_token_id": token,
        }
        for condition, token in (("BASE", 10), ("REMOVED", 11), ("BACKEND_REMOVED", 10), ("CODEC_REMOVED", 10))
    ]
    summary = _summarize(rows, evaluation)
    assert summary["removal_conditions_equal_base"] is False


def test_r13_config_is_json_serializable() -> None:
    assert json.loads(json.dumps(_config())) == _config()


def test_r13_probability_validation_rejects_non_probability_vector() -> None:
    with pytest.raises(R13Error, match="probabilities invalid"):
        _probabilities([0.0] * 8)
