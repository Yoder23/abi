import torch

from abi.capability_compiler_phase4_clarification_route import (
    CLARIFICATION_ROUTE,
    LEGACY_ROUTES,
    NEW_TRAINABLE_PARAMETERS,
    RANK,
    WIDTH,
    ClarificationRouteResidual,
    _schedule,
)


def _inherited():
    generator = torch.Generator().manual_seed(17)
    return {
        "norm.weight": torch.randn(WIDTH, generator=generator),
        "norm.bias": torch.randn(WIDTH, generator=generator),
        "down": torch.randn(LEGACY_ROUTES, RANK, WIDTH, generator=generator),
        "up": torch.randn(LEGACY_ROUTES, WIDTH, RANK, generator=generator),
    }


def test_fifth_route_initialization_preserves_legacy_and_is_zero():
    inherited = _inherited()
    residual = ClarificationRouteResidual(inherited, 155921)
    state = residual.package_state()
    assert sum(value.numel() for value in residual.parameters()) == NEW_TRAINABLE_PARAMETERS
    assert torch.equal(state["norm.weight"], inherited["norm.weight"])
    assert torch.equal(state["norm.bias"], inherited["norm.bias"])
    assert torch.equal(state["down"][:LEGACY_ROUTES], inherited["down"])
    assert torch.equal(state["up"][:LEGACY_ROUTES], inherited["up"])
    hidden = torch.randn(2, 3, WIDTH)
    delta = residual.delta(hidden, torch.tensor([CLARIFICATION_ROUTE, CLARIFICATION_ROUTE]))
    assert torch.equal(delta, torch.zeros_like(delta))


def test_schedule_has_exact_epoch_coverage():
    examples = [{"record_id": f"r{index:03d}"} for index in range(200)]
    scheduled = _schedule(examples, 155921, 10)
    assert len(scheduled) == 2000
    assert {row["record_id"] for row in scheduled} == {row["record_id"] for row in examples}
    assert all(sum(item["record_id"] == row["record_id"] for item in scheduled) == 10 for row in examples)
