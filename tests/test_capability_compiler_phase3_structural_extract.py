import torch

from abi.capability_compiler_phase3_structural_extract import stable_top_indices, target_gate_up, target_o, target_qkv


def test_stable_selection_and_structural_tensor_geometry() -> None:
    scores = torch.tensor([1.0, 4.0, 4.0, 2.0])
    ranked, ordered, margin = stable_top_indices(scores, 2)
    assert ranked.tolist() == [1, 2]
    assert ordered.tolist() == [1, 2]
    assert margin == 2.0
    residual = torch.tensor([1, 3])
    heads = torch.tensor([0, 1])
    qkv = torch.arange(3 * 4 * 4, dtype=torch.float32).view(12, 4)
    assert target_qkv(qkv, residual, heads, source_width=4, head_dim=2, scale=2.0).shape == (12, 2)
    output = torch.arange(4 * 4, dtype=torch.float32).view(4, 4)
    assert target_o(output, residual, heads, head_dim=2, scale=2.0).shape == (2, 4)
    gate_up = torch.arange(2 * 5 * 4, dtype=torch.float32).view(10, 4)
    neurons = torch.tensor([0, 3])
    assert target_gate_up(gate_up, residual, neurons, source_intermediate=5, scale=2.0).shape == (4, 2)
