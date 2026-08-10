import torch

from abi.capability_compiler_phase3_weak_residual import (
    EXPECTED_PARAMETERS,
    SharedWeakResidual,
    WeakBalancedSampler,
    _hook,
    _parameter_count,
)


class _Block:
    pass


def test_parameter_count_is_locked():
    assert _parameter_count(SharedWeakResidual()) == EXPECTED_PARAMETERS


def test_strong_route_bypass_is_bit_exact():
    residual = SharedWeakResidual()
    block = _Block()
    block._abi_weak_capability_routes = torch.tensor([-1, -1])
    hidden = torch.randn(2, 7, 768)
    returned, kwargs = _hook(residual)(block, (hidden,), {"flag": True})
    assert returned[0] is hidden
    assert torch.equal(returned[0], hidden)
    assert kwargs == {"flag": True}


def test_active_route_can_change_hidden_without_touching_bypass_row():
    residual = SharedWeakResidual()
    with torch.no_grad():
        residual.up.weight.fill_(0.01)
        residual.route_shift.weight[1].fill_(0.1)
    block = _Block()
    block._abi_weak_capability_routes = torch.tensor([-1, 1])
    hidden = torch.randn(2, 5, 768)
    returned, _ = _hook(residual)(block, (hidden,), {})
    assert torch.equal(returned[0][0], hidden[0])
    assert not torch.equal(returned[0][1], hidden[1])


def test_weak_sampler_is_balanced():
    rows = [
        {"capability": capability, "index": index}
        for capability in (
            "abstention",
            "coherence",
            "fluent_realization",
            "tone_control",
        )
        for index in range(3)
    ]
    selected = WeakBalancedSampler(rows, 42).batch(12)
    counts = {name: 0 for name in {row["capability"] for row in rows}}
    for row in selected:
        counts[row["capability"]] += 1
    assert set(counts.values()) == {3}
