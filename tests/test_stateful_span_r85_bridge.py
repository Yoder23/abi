from __future__ import annotations

import pytest
import torch

from experiments.stateful_span_r85.core_v1 import (
    MAX_SPAN_TOKENS,
    StatefulSpanBridge,
    SpanPackageError,
    bridge_parameter_count,
)


def test_r85_bridge_contract_masks_padding_and_is_recomputable() -> None:
    torch.manual_seed(85_001)
    bridge = StatefulSpanBridge().eval()
    hidden = torch.randn(2, 7, 768)
    attention = torch.tensor(
        [[True, True, True, True, True, False, False], [True] * 7]
    )
    routes = torch.tensor([12, 12])

    start_logits, length_logits = bridge(hidden, attention, routes)

    assert start_logits.shape == (2, 7)
    assert length_logits.shape == (2, MAX_SPAN_TOKENS)
    assert torch.isneginf(start_logits[0, 5:]).all()
    assert torch.isfinite(start_logits[0, :5]).all()
    assert torch.isfinite(length_logits).all()
    assert bridge_parameter_count() == sum(
        parameter.numel() for parameter in bridge.parameters()
    )


def test_r85_bridge_rejects_contract_drift() -> None:
    bridge = StatefulSpanBridge().eval()
    with pytest.raises(SpanPackageError, match="contract changed"):
        bridge(
            torch.zeros(1, 257, 768),
            torch.ones(1, 257, dtype=torch.bool),
            torch.tensor([12]),
        )
