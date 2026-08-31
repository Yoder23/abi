from __future__ import annotations

import torch

from experiments.foreign_teacher_r12.extractor import extractor_spec
from experiments.foreign_teacher_r12.public_preflight import _atomic_rows
from experiments.native_isa_r11.core import RecurrentTransitionNeuralISA
from experiments.native_transfer_r8.capability_generator import public_capabilities


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
