import torch

from abi.capability_compiler_phase3_shared_gradient_conflict import cosine


def test_cosine_identifies_aligned_orthogonal_and_conflicting_vectors():
    left = torch.tensor([1.0, 0.0])
    assert cosine(left, left) == 1.0
    assert cosine(left, torch.tensor([0.0, 1.0])) == 0.0
    assert cosine(left, -left) == -1.0


def test_zero_vector_is_reported_neutrally():
    assert cosine(torch.zeros(2), torch.ones(2)) == 0.0
