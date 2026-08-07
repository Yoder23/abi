from pathlib import Path

import torch

from abi.capability_compiler_phase3_segment_router import (
    SegmentRouter,
    _semantic_segments,
)


def test_semantic_segments_preserve_metadata_and_body() -> None:
    assert _semantic_segments("metadata\nbody line 1\nbody line 2") == [
        "metadata",
        "body line 1\nbody line 2",
    ]
    assert _semantic_segments("single line") == ["single line"]


def test_segment_router_shapes_and_padding() -> None:
    model = SegmentRouter(20, 8, 4, 6, (1, 2, 3, 5), 15, 0.0)
    ids = torch.tensor(
        [[1, 2, 3, 4, 5, 20], [6, 7, 20, 20, 20, 20]], dtype=torch.long
    )
    lengths = torch.tensor([5, 2], dtype=torch.long)
    assert model(ids, lengths).shape == (2, 15)


def test_protocol_exists() -> None:
    # Added by the preregistration commit after bindings are calculated.
    assert (Path(__file__).parents[1] / "ABI_CAPABILITY_COMPILER_PHASE3_SEGMENT_ROUTER_PROTOCOL_V44.json").is_file()
