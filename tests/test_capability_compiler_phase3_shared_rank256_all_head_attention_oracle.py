import torch

from abi.capability_compiler_phase3_shared_rank256_all_head_attention_oracle import FactoredLinear, stable_factor


def test_full_rank_factor_replays_matrix_equation():
    weight=torch.tensor([[2.0,0.0],[0.0,1.0],[1.0,1.0]])
    output_factor,input_factor,energy=stable_factor(weight,1)
    assert output_factor.shape==(3,1);assert input_factor.shape==(1,2);assert 0.0<energy<1.0


def test_factored_linear_uses_two_physical_projections():
    output_factor=torch.tensor([[1.0],[2.0]])
    input_factor=torch.tensor([[3.0,4.0]])
    layer=FactoredLinear(output_factor,input_factor)
    result=layer(torch.tensor([[1.0,2.0]]))
    torch.testing.assert_close(result,torch.tensor([[11.0,22.0]]))
