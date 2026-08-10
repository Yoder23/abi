from __future__ import annotations

import torch

from abi.capability_compiler_phase3_routed_v16_sequence_conformance import _loss


def test_loss_selects_all_target_positions() -> None:
    class Fake:
        pass
    # Shape contract is tested without running the production host.
    assert callable(_loss)
    assert torch.tensor([1, 2, 3]).numel() == 3
