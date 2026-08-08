import torch

from abi.capability_compiler_phase3_lexical_substrate_extract import project_rows


def test_projection_uses_declared_row_norm():
    rows = torch.eye(3)
    projection = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = project_rows(rows, projection, 2.0)
    assert result.dtype == torch.float16
    assert torch.allclose(result.float().norm(dim=1), torch.full((3,), 2.0), atol=0.002)
