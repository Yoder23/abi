"""Fixed behavioral compiler from a frozen foreign teacher into the R11 ABI."""

from __future__ import annotations

import hashlib
from typing import Any

import torch

from experiments.native_isa_r11.core import R11Error
from experiments.native_transfer_r8.capability_generator import (
    MODULUS,
    OPERATORS,
    canonical_json_bytes,
    render_prompt,
)
from experiments.native_transfer_r8.native_host import FrozenNeuralHost

EXTRACTOR_FORMAT = "abi-r12a-foreign-teacher-atomic-compiler/1"


def extractor_spec() -> dict[str, Any]:
    value = {
        "format": EXTRACTOR_FORMAT,
        "family": "opaque_modular_micro_language/1",
        "source_interface": "frozen_teacher_native_full_vocabulary_logits",
        "probe_order": [
            {"operation_slot": operation, "input_slot": start}
            for operation in range(len(OPERATORS))
            for start in range(MODULUS)
        ],
        "canonicalization": "native_canonical_token_argmax_to_one_hot_float32",
        "probe_count": len(OPERATORS) * MODULUS,
        "learned_parameters": 0,
        "training_rows_accessed": 0,
        "evaluation_rows_accessed": 0,
        "answers_accessed": 0,
        "capability_rule_accessed": False,
    }
    value["extractor_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


@torch.inference_mode()
def extract_transition(host: FrozenNeuralHost) -> tuple[torch.Tensor, dict[str, Any]]:
    prompts = [
        render_prompt(start, (operation,))
        for operation in range(len(OPERATORS))
        for start in range(MODULUS)
    ]
    logits, _ = host.logits(prompts, prefix=None)
    native_predictions = logits.argmax(dim=-1)
    canonical_ids = torch.tensor(host.target_token_ids, device=logits.device)
    canonical_logits = logits.index_select(-1, canonical_ids)
    canonical_predictions = canonical_logits.argmax(dim=-1)
    expected_native = canonical_ids.index_select(0, canonical_predictions)
    if not torch.equal(native_predictions, expected_native):
        raise R11Error("foreign teacher atomic output escapes canonical ABI tokens")
    transition = torch.nn.functional.one_hot(canonical_predictions, num_classes=MODULUS).float()
    transition = transition.reshape(len(OPERATORS), MODULUS, MODULUS).cpu().contiguous()
    receipt = {
        "extractor": extractor_spec(),
        "native_prediction_token_ids": [int(value) for value in native_predictions.cpu().tolist()],
        "canonical_predictions": [int(value) for value in canonical_predictions.cpu().tolist()],
        "transition_shape": list(transition.shape),
        "transition_nonzero_values": int(torch.count_nonzero(transition)),
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    return transition, receipt
