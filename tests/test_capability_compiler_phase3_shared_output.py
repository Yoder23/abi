import torch

from abi.capability_compiler_phase3_shared_output import (
    EXPECTED_TRAINABLE_PARAMETERS,
    SharedOutputCake,
    wrong_recent_repeat_margin_loss,
)
from abi.capability_compiler_phase3_sequence_bridge import PromptConditionedSequenceBridge


def test_shared_output_parameter_contract_and_identity_install():
    bridge = PromptConditionedSequenceBridge()
    classifier = torch.nn.Linear(128, 6)
    cake = SharedOutputCake()
    count = sum(p.numel() for p in bridge.parameters()) + sum(p.numel() for p in classifier.parameters()) + sum(p.numel() for p in cake.parameters())
    assert count == EXPECTED_TRAINABLE_PARAMETERS
    hidden = torch.randn(2, 4, 768)
    assert torch.equal(cake(hidden), hidden)


def test_wrong_recent_repeat_loss_targets_only_wrong_recent_argmax():
    logits = torch.zeros(1, 5, 8)
    labels = torch.tensor([[-100, 1, 2, 3, 4]])
    logits[0, 2, 1] = 5.0
    logits[0, 2, 3] = 1.0
    loss, events = wrong_recent_repeat_margin_loss(logits, labels, window=4, margin=0.5)
    assert events == 1
    assert loss.item() > 4.0
    logits[0, 2, 3] = 6.0
    loss, events = wrong_recent_repeat_margin_loss(logits, labels, window=4, margin=0.5)
    assert events == 0
    assert loss.item() == 0.0
