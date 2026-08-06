import torch

from abi.capability_compiler_phase3_repeat_prefix import construct_recent_repeat_batch


def test_recent_repeat_corrupts_objective_repeat_only():
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    labels = torch.tensor([[-100, 11, 12, 13, 14, 15]])
    logits = torch.full((1, 6, 32), -10.0)
    logits[0, 0, 9] = 10.0   # wrong but not a prior response token: ignored
    logits[0, 1, 11] = 10.0  # wrong and repeats target 11: selected
    corrupted, recovery, events = construct_recent_repeat_batch(ids, labels, logits, horizon=2)
    assert events == 1
    assert corrupted.tolist() == [[10, 11, 11, 13, 14, 15]]
    assert recovery.tolist() == [[-100, -100, -100, 13, 14, -100]]


def test_recent_repeat_leaves_valid_alternative_mismatch_untrained():
    ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, 2, 3, 4]])
    logits = torch.full((1, 4, 8), -10.0)
    logits[0, 0, 7] = 10.0
    logits[0, 1, 6] = 10.0
    corrupted, recovery, events = construct_recent_repeat_batch(ids, labels, logits, horizon=16)
    assert events == 0
    assert torch.equal(corrupted, ids)
    assert bool((recovery == -100).all())
