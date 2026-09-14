from __future__ import annotations

import pytest
import torch

from experiments.joint_span_r88.core_v1 import (
    MAX_SPAN_TOKENS,
    JointSpanBridge,
    JointSpanPackageError,
    bridge_parameter_count,
    joint_span_scores,
)


def test_r88_joint_span_masks_padding_direction_and_length() -> None:
    torch.manual_seed(88_001)
    bridge = JointSpanBridge().eval()
    hidden = torch.randn(2, 20, 768)
    attention = torch.tensor([[True] * 17 + [False] * 3, [True] * 20])
    routes = torch.tensor([12, 12])
    start, end = bridge(hidden, attention, routes)
    scores = joint_span_scores(start, end, attention)

    assert scores.shape == (2, 20, 20)
    assert torch.isneginf(scores[0, :, 17:]).all()
    assert torch.isneginf(scores[0, 7, 6])
    assert torch.isneginf(scores[0, 0, MAX_SPAN_TOKENS])
    assert torch.isfinite(scores[0, 7, 7])
    assert bridge_parameter_count() == sum(
        parameter.numel() for parameter in bridge.parameters()
    )


def test_r88_bridge_rejects_contract_drift() -> None:
    bridge = JointSpanBridge().eval()
    with pytest.raises(JointSpanPackageError, match="contract changed"):
        bridge(
            torch.zeros(1, 257, 768),
            torch.ones(1, 257, dtype=torch.bool),
            torch.tensor([12]),
        )
