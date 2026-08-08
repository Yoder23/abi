import torch

from abi.capability_compiler_phase3_lexical_substrate_core import initialize_substrate


class _Model:
    def __init__(self):
        self.lexeme_embedding = torch.nn.Embedding(32015, 192)
        self.fixed_output = torch.nn.Linear(192, 32015)


def test_initialize_substrate_is_exact_and_frozen():
    model = _Model()
    substrate = {"input_embedding_rows_fp16": torch.randn(32011, 192).half(), "output_head_rows_fp16": torch.randn(32011, 192).half()}
    initialize_substrate(model, substrate)
    assert torch.equal(model.lexeme_embedding.weight[4:], substrate["input_embedding_rows_fp16"].float())
    assert torch.equal(model.fixed_output.weight[4:], substrate["output_head_rows_fp16"].float())
    assert not model.lexeme_embedding.weight.requires_grad
    assert not model.fixed_output.weight.requires_grad
