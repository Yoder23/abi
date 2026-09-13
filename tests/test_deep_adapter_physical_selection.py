from __future__ import annotations

import torch

from abi.layercake_core_loader import _make_deep_adapter_hook


class _Adapter(torch.nn.Module):
    def __init__(self, width: int, rank: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(width)
        self.down = torch.nn.Linear(width, rank, bias=False)
        self.up = torch.nn.Linear(rank, width, bias=False)


class _Block:
    pass


def _historical(adapters, hidden, routes):
    norm_weight = torch.stack([adapter.norm.weight for adapter in adapters]).index_select(0, routes)
    norm_bias = torch.stack([adapter.norm.bias for adapter in adapters]).index_select(0, routes)
    down_weight = torch.stack([adapter.down.weight for adapter in adapters]).index_select(0, routes)
    up_weight = torch.stack([adapter.up.weight for adapter in adapters]).index_select(0, routes)
    mean = hidden.mean(dim=-1, keepdim=True)
    variance = (hidden - mean).square().mean(dim=-1, keepdim=True)
    normalized = (hidden - mean) * torch.rsqrt(variance + 1.0e-5)
    normalized = normalized * norm_weight[:, None] + norm_bias[:, None]
    low = torch.bmm(normalized, down_weight.transpose(1, 2))
    return hidden + torch.bmm(torch.nn.functional.silu(low), up_weight.transpose(1, 2))


def test_uniform_route_selects_one_adapter_and_is_numerically_equivalent() -> None:
    torch.manual_seed(74)
    adapters = torch.nn.ModuleList(_Adapter(16, 4) for _ in range(14))
    hidden = torch.randn(3, 7, 16)
    routes = torch.full((3,), 12, dtype=torch.long)
    block = _Block()
    block._abi_selected_capability_routes = routes
    actual = _make_deep_adapter_hook(adapters)(block, (hidden,), {})[0][0]
    expected = _historical(adapters, hidden, routes)
    assert torch.equal(actual, expected)
    assert block._abi_last_deep_adapter_routes == (12,)


def test_mixed_route_training_path_records_only_observed_routes() -> None:
    torch.manual_seed(75)
    adapters = torch.nn.ModuleList(_Adapter(16, 4) for _ in range(14))
    hidden = torch.randn(3, 7, 16)
    routes = torch.tensor([2, 12, 2], dtype=torch.long)
    block = _Block()
    block._abi_selected_capability_routes = routes
    actual = _make_deep_adapter_hook(adapters)(block, (hidden,), {})[0][0]
    assert torch.equal(actual, _historical(adapters, hidden, routes))
    assert block._abi_last_deep_adapter_routes == (2, 12)
