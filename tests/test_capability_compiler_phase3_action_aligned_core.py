import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_action_aligned_core import _cosine_loss


def test_action_alignment_loss_is_masked_and_directional() -> None:
    student = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    teacher = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    assert float(_cosine_loss(student, teacher, torch.tensor([[True, False]]))) == pytest.approx(0.0)
    assert float(_cosine_loss(student, teacher, torch.tensor([[True, True]]))) == pytest.approx(0.5)
    with pytest.raises(Phase3Error):
        _cosine_loss(student, teacher, torch.tensor([[False, False]]))
