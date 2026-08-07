from pathlib import Path

import torch

from abi.capability_compiler_phase3_sparse_router import (
    SparseRouter,
    _character_features,
)


def test_character_features_are_deterministic_and_casefolded() -> None:
    first = _character_features("Logical Order", 1024, 2, 5, 450045)
    second = _character_features("logical   order", 1024, 2, 5, 450045)
    assert first == second
    assert first
    assert all(0 <= value < 1024 for value in first)


def test_sparse_router_shapes() -> None:
    model = SparseRouter(20, 32, 15)
    bpe_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    offsets = torch.tensor([0, 2], dtype=torch.long)
    character_ids = torch.tensor([5, 6, 7, 8, 9], dtype=torch.long)
    character_offsets = torch.tensor([0, 3], dtype=torch.long)
    assert model(bpe_ids, offsets, character_ids, character_offsets).shape == (2, 15)


def test_protocol_exists() -> None:
    assert (Path(__file__).parents[1] / "ABI_CAPABILITY_COMPILER_PHASE3_SPARSE_ROUTER_PROTOCOL_V45.json").is_file()
