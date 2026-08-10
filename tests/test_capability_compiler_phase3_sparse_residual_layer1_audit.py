import torch
from abi.capability_compiler_phase3_sparse_residual_layer1_audit import balanced_recursive_pca, deterministic_kmeans


def test_deterministic_kmeans_is_stable_and_nonempty() -> None:
    values = torch.tensor([[0.0], [0.1], [9.9], [10.0]])
    first_labels, first_centers = deterministic_kmeans(values, 2, 4)
    second_labels, second_centers = deterministic_kmeans(values, 2, 4)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(first_centers, second_centers)
    assert sorted(torch.bincount(first_labels).tolist()) == [2, 2]


def test_balanced_recursive_pca_is_exact_and_deterministic() -> None:
    values = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    first_labels, first_centers = balanced_recursive_pca(values, 4)
    second_labels, second_centers = balanced_recursive_pca(values, 4)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(first_centers, second_centers)
    assert torch.bincount(first_labels).tolist() == [4, 4, 4, 4]
