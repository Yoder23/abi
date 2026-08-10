import torch

from abi.capability_compiler_phase3_mlp_residual_rank_audit import (
    centered_covariance,
    project_with_basis,
)


def test_centered_covariance_and_projection_are_exact_for_known_basis() -> None:
    values = torch.tensor(
        [[1.0, 2.0, 9.0], [2.0, 4.0, 9.0], [3.0, 6.0, 9.0]],
        dtype=torch.float32,
    )
    mean, covariance, observations = centered_covariance([values], 3, torch.device("cpu"))
    assert observations == 3
    assert torch.allclose(mean, torch.tensor([2.0, 4.0, 9.0]))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -1:]
    reconstructed = project_with_basis(values, mean, basis)
    assert torch.allclose(reconstructed, values, atol=1e-5, rtol=1e-5)
