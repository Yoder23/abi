import torch

from abi.capability_compiler_phase3_labeled_bpe_core import _previous_actions, use_body_view


def test_header_view_is_deterministic():
    assert use_body_view("record-1", 7, 0.5) == use_body_view("record-1", 7, 0.5)
    assert not use_body_view("record-1", 7, 0.0)
    assert use_body_view("record-1", 7, 1.0)


def test_history_corruption_never_changes_bos_or_padding():
    targets = torch.tensor([[4, 5, 2, -100]])
    previous, corrupted, eligible = _previous_actions(targets, 1.0)
    assert previous.tolist() == [[1, 3, 3, 3]]
    assert corrupted == eligible == 3
