from __future__ import annotations

import torch

from abi.layercake_transformer_student_control import disable_task_cake_effects


class _Cake(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = torch.nn.Linear(3, 4, bias=False)


class _TinyControl(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_cakes = torch.nn.ModuleList([_Cake(), _Cake()])
        self.task_classifier = torch.nn.Linear(4, 2)


def test_disable_task_cake_effects_is_exact_and_preserves_classifier():
    model = _TinyControl()
    classifier_before = {
        name: tensor.detach().clone()
        for name, tensor in model.task_classifier.state_dict().items()
    }
    result = disable_task_cake_effects(model)
    assert result["identity_task_cakes"] == 2
    assert result["zeroed_up_projection_parameters"] == 24
    assert result["task_cake_effect_disabled_exact"] is True
    assert all(
        torch.count_nonzero(cake.up.weight).item() == 0
        for cake in model.task_cakes
    )
    assert all(
        torch.equal(model.task_classifier.state_dict()[name], tensor)
        for name, tensor in classifier_before.items()
    )
