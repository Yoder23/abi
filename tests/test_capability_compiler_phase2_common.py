from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from abi.capability_compiler_phase2_common import (
    CAPABILITIES,
    CompactTransformerLM,
    capture_lora,
    evaluate_functional,
    install_lora,
    load_lora,
    lora_modules,
    repetition_collapse,
    reset_lora,
    state_sha256,
)
from abi.capability_compiler_phase2_lora import route_prompt, train_router
from abi.capability_compiler_phase2_analysis import (
    stratified_paired_bootstrap,
    validate_output_suite,
    wilson,
)


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(2, 6, bias=False)
        self.o_proj = nn.Linear(2, 2, bias=False)


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(2, 8, bias=False)
        self.down_proj = nn.Linear(4, 2, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()


class _PhiGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer() for _ in range(32)])


def test_internal_lora_target_graph_and_state_round_trip():
    model = _PhiGraph()
    targets = install_lora(model, rank=2, alpha=4.0, dropout=0.05)
    assert len(targets) == 128
    assert len(lora_modules(model)) == 128
    reset_lora(model, seed=104729, capability="grammar")
    state = capture_lora(model)
    digest = state_sha256(state)
    reset_lora(model, seed=130363, capability="grammar")
    assert state_sha256(capture_lora(model)) != digest
    load_lora(model, state)
    assert state_sha256(capture_lora(model)) == digest


def test_compact_student_is_within_two_percent_of_locked_active_byte_envelope():
    model = CompactTransformerLM()
    assert model.parameter_count == 11_060_800
    assert (model.parameter_count * 2) / 21_720_964 < 1.02
    logits = model(torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert logits.shape == (1, 3, 32_064)


def test_functional_evaluators_and_repetition_detection():
    assert evaluate_functional("Mira walks Monday", {"kind": "contains_all", "values": ["Mira", "walks", "Monday"]})
    assert evaluate_functional("A then B then C", {"kind": "ordered_contains", "values": ["A", "B", "C"]})
    assert evaluate_functional("item: meeting note\ncode: N1", {"kind": "regex", "pattern": r"^item: meeting note\ncode: N1$"})
    assert evaluate_functional("exact", {"kind": "exact", "value": "exact"})
    assert evaluate_functional("ExAcT", {"kind": "exact", "value": "exact"})
    assert repetition_collapse("echo " * 10)
    assert not repetition_collapse("one two three four five six seven eight")


def test_router_is_trained_only_from_declared_acquisition_labels():
    rows = []
    for capability in CAPABILITIES:
        rows.extend({"capability": capability, "normalized_acquisition_prompt": f"{capability} marker {index}"} for index in range(500))
    centroids = train_router(rows)
    assert set(centroids) == set(CAPABILITIES)
    assert route_prompt("grammar marker new", centroids) == "grammar"
    assert sum(value.size for value in centroids.values()) == 14 * 4096


def test_paired_statistics_are_stratified_and_deterministic():
    reference = []
    candidate = []
    for capability in CAPABILITIES:
        for index in range(100):
            base = {
                "probe_id": f"{capability}-{index}",
                "capability": capability,
                "repetition_collapse": False,
                "output": "ok",
                "output_token_ids": [1],
            }
            reference.append({**base, "functional_pass": index < 80})
            candidate.append({**base, "functional_pass": index < 90})
    validate_output_suite(reference, expected_per_capability=100)
    result = stratified_paired_bootstrap(candidate, reference, resamples=1_000, seed=1729)
    assert result["observed_difference"] == pytest.approx(0.1)
    assert result == stratified_paired_bootstrap(candidate, reference, resamples=1_000, seed=1729)
    lower, upper = wilson(90, 100)
    assert lower < 0.9 < upper
