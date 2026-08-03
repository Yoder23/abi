from __future__ import annotations

import pytest
import torch

from abi.layercake_component_graft import ComponentGraftError, graft_state_dict


def test_graft_copies_only_sparse_cakes_and_router() -> None:
    base = {
        "transformer.weight": torch.tensor([1.0, 2.0]),
        "task_cakes.0.down.weight": torch.tensor([3.0]),
        "task_classifier.weight": torch.tensor([4.0]),
    }
    donor = {
        "transformer.weight": torch.tensor([9.0, 9.0]),
        "task_cakes.0.down.weight": torch.tensor([5.0]),
        "task_classifier.weight": torch.tensor([6.0]),
    }

    result, selected, changed = graft_state_dict(base, donor)

    assert torch.equal(result["transformer.weight"], base["transformer.weight"])
    assert torch.equal(
        result["task_cakes.0.down.weight"],
        donor["task_cakes.0.down.weight"],
    )
    assert torch.equal(
        result["task_classifier.weight"], donor["task_classifier.weight"]
    )
    assert selected == ["task_cakes.0.down.weight", "task_classifier.weight"]
    assert changed == selected


def test_graft_rejects_incompatible_names() -> None:
    with pytest.raises(ComponentGraftError, match="tensor names differ"):
        graft_state_dict(
            {"task_classifier.weight": torch.tensor([1.0])},
            {"task_classifier.bias": torch.tensor([1.0])},
        )


def test_graft_rejects_incompatible_shape() -> None:
    with pytest.raises(ComponentGraftError, match="incompatible tensor"):
        graft_state_dict(
            {"task_classifier.weight": torch.tensor([1.0])},
            {"task_classifier.weight": torch.tensor([1.0, 2.0])},
        )
