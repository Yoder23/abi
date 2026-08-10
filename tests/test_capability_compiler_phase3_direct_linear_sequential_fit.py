import torch

from abi.capability_compiler_phase3_direct_linear_sequential_fit import key_classes


class _Fake(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.layers[0]
        layer.input_norm = torch.nn.Linear(1, 1, bias=False)
        layer.attention_input_projection = torch.nn.Linear(1, 1, bias=False)
        layer.attention_norm = torch.nn.Linear(1, 1, bias=False)
        layer.qkv_proj = torch.nn.Linear(1, 1, bias=False)
        layer.o_proj = torch.nn.Linear(1, 1, bias=False)
        layer.attention_output_projection = torch.nn.Linear(1, 1, bias=False)
        layer.post_attention_norm = torch.nn.Linear(1, 1, bias=False)
        layer.mlp_coefficient_projection = torch.nn.Linear(1, 1, bias=False)
        layer.mlp_output_projection = torch.nn.Linear(1, 1, bias=False)
        layer.mlp_residual_mean = torch.nn.Parameter(torch.zeros(1))


def test_exact_attention_and_analytic_import_boundaries() -> None:
    attention, imported = key_classes(_Fake())
    assert attention == {
        "layers.0.attention_input_projection.weight",
        "layers.0.attention_norm.weight",
        "layers.0.qkv_proj.weight",
        "layers.0.o_proj.weight",
        "layers.0.attention_output_projection.weight",
    }
    assert imported == {
        "layers.0.mlp_coefficient_projection.weight",
        "layers.0.mlp_output_projection.weight",
        "layers.0.mlp_residual_mean",
    }
