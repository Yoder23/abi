from __future__ import annotations

from pathlib import Path

import pytest
import torch

from experiments.native_isa_r11.core import (
    R11Error,
    RecurrentTransitionNeuralISA,
    load_package,
    train_teacher_transition,
    transition_accuracy,
    write_package_once,
)
from experiments.native_transfer_r8.capability_generator import (
    generate_rows,
    public_capabilities,
)


def _capability_rows() -> tuple[object, list[dict], list[dict]]:
    capability = public_capabilities(90210, split="development", count=1)[0]
    training = generate_rows(
        capability,
        split="r11_test_train",
        rows=288,
        depths=[1, 2, 3],
        seed=41,
    )
    evaluation = generate_rows(
        capability,
        split="r11_test_evaluation",
        rows=128,
        depths=[4, 5, 6, 7],
        seed=43,
    )
    return capability, training, evaluation


def test_teacher_learning_generalizes_and_package_roundtrips(tmp_path: Path) -> None:
    _, training, evaluation = _capability_rows()
    transition, receipt = train_teacher_transition(
        training, steps=300, learning_rate=0.1, seed=17, device="cpu"
    )
    assert receipt["trainable_parameters"] == 192
    assert transition_accuracy(transition, evaluation) == 1.0
    item = write_package_once(
        tmp_path,
        transition,
        {"teacher_before_sha256": "a" * 64, "teacher_after_sha256": "b" * 64},
    )
    _, restored = load_package(tmp_path / item["path"])
    assert torch.equal(transition, restored)
    probabilities = RecurrentTransitionNeuralISA()(restored, [evaluation[0]["prompt"]])
    assert int(probabilities.argmax(dim=-1)) == evaluation[0]["answer"]


def test_native_package_rejects_tampering(tmp_path: Path) -> None:
    _, training, _ = _capability_rows()
    transition, _ = train_teacher_transition(
        training, steps=50, learning_rate=0.1, seed=19, device="cpu"
    )
    item = write_package_once(
        tmp_path,
        transition,
        {"teacher_before_sha256": "a" * 64, "teacher_after_sha256": "b" * 64},
    )
    path = tmp_path / item["path"]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 1
    path.write_bytes(data)
    with pytest.raises(R11Error):
        load_package(path)
