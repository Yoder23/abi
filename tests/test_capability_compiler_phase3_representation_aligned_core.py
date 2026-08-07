import torch

from abi.capability_compiler_phase3_representation_aligned_core import (
    _masked_mean,
    _projection,
    _projection_hash,
)


def test_fixed_projection_is_deterministic_and_seed_sensitive() -> None:
    first = _projection(12, 4, 62)
    second = _projection(12, 4, 62)
    third = _projection(12, 4, 63)
    assert torch.equal(first, second)
    assert _projection_hash(first) == _projection_hash(second)
    assert not torch.equal(first, third)
    assert torch.allclose(first.norm(dim=0), torch.ones(4), atol=1e-6)


def test_masked_mean_excludes_padding() -> None:
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    valid = torch.tensor([[True, True, False]])
    assert torch.equal(_masked_mean(value, valid), torch.tensor([[2.0, 3.0]]))
