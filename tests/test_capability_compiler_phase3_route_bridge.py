import torch
from abi.capability_compiler_phase3_route_bridge import _encoded


class IdentityEncoder(torch.nn.Module):
    def forward(self, value, **_kwargs):
        return value


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.lexeme_embedding=torch.nn.Embedding(8,4,padding_idx=0);self.source_position=torch.nn.Embedding(3,4);self.encoder=IdentityEncoder()


def test_route_embedding_replaces_only_first_source_embedding() -> None:
    model=Tiny();routes=torch.nn.Embedding(2,4);source=torch.tensor([[5,6,0]]);labels=torch.tensor([1]);encoded,padding=_encoded(model,source,routes,labels)
    assert torch.equal(encoded[0,0],routes(labels)[0]+model.source_position(torch.tensor(0)))
    assert padding.tolist()==[[False,False,True]]
