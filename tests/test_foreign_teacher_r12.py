from __future__ import annotations

import hashlib

import pytest
import torch

from experiments.foreign_teacher_r12.extractor import extractor_spec
from experiments.foreign_teacher_r12.public_preflight import _atomic_rows
from experiments.foreign_teacher_r12.teacher import R12TeacherError
from experiments.foreign_teacher_r12.verify_public import _evidence
from experiments.native_isa_r11.core import RecurrentTransitionNeuralISA
from experiments.native_transfer_r8.capability_generator import (
    canonical_json_bytes,
    public_capabilities,
)


def test_r12_extractor_is_fixed_and_has_no_answer_inputs() -> None:
    spec = extractor_spec()
    assert spec["probe_count"] == 24
    assert spec["learned_parameters"] == 0
    assert spec["training_rows_accessed"] == 0
    assert spec["evaluation_rows_accessed"] == 0
    assert spec["answers_accessed"] == 0
    assert spec["capability_rule_accessed"] is False


def test_atomic_probe_order_can_define_an_exact_r11_transition() -> None:
    capability = public_capabilities(12012001, split="development", count=1)[0]
    rows = _atomic_rows(capability)
    predictions = torch.tensor([row["answer"] for row in rows])
    transition = torch.nn.functional.one_hot(predictions, num_classes=8).float()
    transition = transition.reshape(3, 8, 8)
    executor = RecurrentTransitionNeuralISA()
    outputs = executor(transition, [row["prompt"] for row in rows]).argmax(dim=-1)
    assert outputs.tolist() == [row["answer"] for row in rows]


def test_public_verifier_accepts_only_recomputable_receipt_hash() -> None:
    receipt = {"format": "example", "measurement": 1}
    receipt["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    _evidence(receipt)

    receipt["measurement"] = 2
    with pytest.raises(R12TeacherError, match="evidence hash changed"):
        _evidence(receipt)


def test_public_verifier_rejects_missing_receipt_hash() -> None:
    with pytest.raises(R12TeacherError, match="evidence hash changed"):
        _evidence({"format": "example"})
