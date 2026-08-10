import torch
from abi.capability_compiler_phase3_sparse_residual_layer1_audit import deterministic_kmeans


def test_deterministic_kmeans_is_stable_and_nonempty() -> None:
    values = torch.tensor([[0.0], [0.1], [9.9], [10.0]])
    first_labels, first_centers = deterministic_kmeans(values, 2, 4)
    second_labels, second_centers = deterministic_kmeans(values, 2, 4)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(first_centers, second_centers)
    assert sorted(torch.bincount(first_labels).tolist()) == [2, 2]
