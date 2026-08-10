import torch

from abi.capability_compiler_phase3_token_substrate_conformance import configure_trainable


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.wpe = torch.nn.Embedding(4, 3)
        self.transformer.wte = torch.nn.Embedding(5, 3)
        self.head = torch.nn.Linear(3, 2)


def test_scope_trains_token_substrate_but_freezes_positions():
    model = Tiny(); configure_trainable(model)
    assert model.transformer.wte.weight.requires_grad
    assert not model.transformer.wpe.weight.requires_grad
    assert model.head.weight.requires_grad
