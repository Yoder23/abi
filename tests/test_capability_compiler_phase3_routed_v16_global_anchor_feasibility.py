from __future__ import annotations

import torch

from abi.capability_compiler_phase3_routed_v16_global_anchor_feasibility import _advance_candidate


def test_zero_layer_advance_is_identity() -> None:
    hidden = torch.randn(1, 3, 4)
    assert _advance_candidate(object(), hidden, torch.arange(3), 0, 0) is hidden
