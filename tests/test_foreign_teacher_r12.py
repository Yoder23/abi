from __future__ import annotations

import hashlib
import json
import random

import pytest
import torch

from experiments.foreign_teacher_r12.custody import verify_r11_freeze
from experiments.foreign_teacher_r12.extractor import extractor_spec
from experiments.foreign_teacher_r12.public_preflight import _atomic_rows
from experiments.foreign_teacher_r12.teacher import (
    R12TeacherError,
    sample_training_batch,
)
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


def test_r11_freeze_recomputes_bindings_and_rejects_tampering(tmp_path) -> None:
    bound = tmp_path / "bound.txt"
    bound.write_text("frozen", encoding="utf-8")
    bound_sha = hashlib.sha256(bound.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"bindings": {"bound.txt": bound_sha}}), encoding="utf-8"
    )
    config = {
        "r11_freeze": {
            "binding_manifest": "manifest.json",
            "binding_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "sealed_tag": "example",
            "sealed_commit": "0" * 40,
        }
    }
    receipt = verify_r11_freeze(tmp_path, config)
    assert receipt["verified_bindings"] == {"bound.txt": bound_sha}

    bound.write_text("changed", encoding="utf-8")
    with pytest.raises(R12TeacherError, match="binding changed"):
        verify_r11_freeze(tmp_path, config)


def test_depth_balanced_teacher_sampling_is_exact() -> None:
    rows = [
        {"depth": depth, "row": index}
        for depth, count in ((1, 2), (2, 5), (3, 11), (4, 17), (5, 23))
        for index in range(count)
    ]
    batch = sample_training_batch(
        rows,
        batch_size=10,
        generator=random.Random(12012004),
        strategy="depth_balanced",
    )
    assert {depth: sum(row["depth"] == depth for row in batch) for depth in range(1, 6)} == {
        depth: 2 for depth in range(1, 6)
    }


def test_depth_balanced_teacher_sampling_rejects_uneven_batch() -> None:
    rows = [{"depth": depth} for depth in range(1, 6)]
    with pytest.raises(R12TeacherError, match="divide evenly"):
        sample_training_batch(
            rows,
            batch_size=8,
            generator=random.Random(1),
            strategy="depth_balanced",
        )
