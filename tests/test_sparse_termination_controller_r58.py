import pytest
import torch

from experiments.sparse_termination_controller_r58.controller import (
    RANK,
    WIDTH,
    SparseTerminationController,
    balanced_terminal_bce,
)


def test_controller_selects_only_supplied_routes():
    controller = SparseTerminationController(seed=7)
    logits = controller(torch.randn(2, 3, WIDTH), torch.tensor([2, 11]))
    assert logits.shape == (2, 3)
    assert controller.last_calls == (2, 11)
    assert sum(parameter.numel() for parameter in controller.parameters()) == 14 * (
        RANK * WIDTH + RANK + RANK + 1
    )


def test_balanced_terminal_bce_requires_one_eos_per_record():
    logits = torch.tensor([[-2.0, -1.0, 2.0]])
    labels = torch.tensor([[-100, 3, 4, 9]])
    loss = balanced_terminal_bce(logits, labels, 9)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="one observable EOS"):
        balanced_terminal_bce(logits, torch.tensor([[-100, 3, 4, 5]]), 9)
