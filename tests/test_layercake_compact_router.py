import torch

from abi.layercake_compact_router import (
    CompactTaskRouter,
    MAX_ROUTER_TOKENS,
    OUTPUT_ROUTES,
    _holdout,
)


def test_compact_router_shapes_and_token_limit():
    model = CompactTaskRouter().eval()
    prompt = torch.arange(MAX_ROUTER_TOKENS + 23).remainder(50_257)[None, :]
    scores, route = model(prompt)
    assert scores.shape == (1, OUTPUT_ROUTES)
    assert route.shape == (1,)
    short_scores, _ = model(prompt[:, :MAX_ROUTER_TOKENS])
    torch.testing.assert_close(scores, short_scores)


def test_compact_router_holdout_is_content_addressed():
    values = [_holdout(f"record-{index}") for index in range(1000)]
    assert values == [_holdout(f"record-{index}") for index in range(1000)]
    assert 70 <= sum(values) <= 130
