import json

import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_sequence_bridge import (
    BRIDGE_RANK,
    EXPECTED_TRAINABLE_PARAMETERS,
    PromptConditionedSequenceBridge,
    SYSTEMS,
    _is_bridge_tensor,
    load_protocol,
)


def test_sequence_bridge_is_identity_at_install_and_prompt_conditioned():
    torch.manual_seed(7)
    bridge = PromptConditionedSequenceBridge()
    hidden = torch.randn(2, 5, 768)
    prompt_lengths = torch.tensor([5, 3])
    summary = bridge.encode(
        hidden,
        prompt_lengths=prompt_lengths,
        attention_mask=None,
    )
    assert summary.shape == (2, BRIDGE_RANK)
    assert not torch.equal(summary[0], summary[1])
    routes = torch.tensor([0, 5])
    conditioned = bridge.conditioned(summary, routes)
    for adapter in bridge.adapters:
        assert torch.equal(adapter(hidden, conditioned), hidden)


def test_sequence_bridge_trainable_parameter_contract_is_small():
    bridge = PromptConditionedSequenceBridge()
    route_classifier = torch.nn.Linear(BRIDGE_RANK, 6)
    bridge_parameters = sum(parameter.numel() for parameter in bridge.parameters())
    classifier_parameters = sum(
        parameter.numel() for parameter in route_classifier.parameters()
    )
    six_rank64_cakes = 6 * (2 * 768 + 2 * 768 * 64)
    assert bridge_parameters == 957_184
    assert classifier_parameters == 774
    assert bridge_parameters + classifier_parameters + six_rank64_cakes == EXPECTED_TRAINABLE_PARAMETERS


def test_registered_scope_rejects_host_and_unregistered_routes():
    assert _is_bridge_tensor("abi_sequence_bridge.adapters.0.up.weight")
    assert _is_bridge_tensor("abi_sequence_route_classifier.weight")
    assert _is_bridge_tensor("task_cakes.5.up.weight")
    assert not _is_bridge_tensor("task_cakes.6.up.weight")
    assert not _is_bridge_tensor("transformer.h.0.attn.c_attn.weight")
    assert SYSTEMS == ("B0", "B1", "B2", "B3", "B4")


def test_protocol_fails_closed_on_governance_mutation(tmp_path):
    protocol = {
        "format": "abi-capability-compiler-phase3-sequence-successor/1",
        "status": "PREREGISTERED_CONDITIONAL_SUCCESSOR",
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "final_test_access": "PROHIBITED",
        "phase3_promotion_eligible": False,
        "systems": {
            "B0": "labeled prompt-conditioned sequence bridge",
            "B1": "label-free prompt-hash routes with the same sequence bridge",
            "B2": "labeled routes with within-capability target derangement",
            "B3": "same bridge and routing supervision with no teacher-response loss",
            "B4": "same prompt-conditioned bridge with one monolithic output route",
        },
        "architecture": {
            "rank": BRIDGE_RANK,
            "trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
            "frozen_transformer_blocks": 3,
            "source_parameters_copied": 0,
        },
        "bindings": {},
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    loaded, _ = load_protocol(tmp_path, path)
    assert loaded["final_test_access"] == "PROHIBITED"

    protocol["final_test_access"] = "ALLOWED"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(tmp_path, path)
