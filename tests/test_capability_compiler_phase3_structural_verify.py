import torch

from abi.capability_compiler_phase3_structural_verify import _select


def test_independent_structural_selection_is_stable_and_ordered() -> None:
    assert _select(torch.tensor([2.0, 5.0, 5.0, 1.0]), 2).tolist() == [1, 2]
