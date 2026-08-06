import torch

from abi.capability_compiler_phase3_self_prefix import construct_self_prefix_batch


def test_construct_self_prefix_batch_uses_first_wrong_and_future_horizon():
    ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
    labels = torch.tensor([[-100, -100, 12, 13, 14, 15]])
    logits = torch.full((1, 6, 32), -10.0)
    logits[0, 1, 9] = 10.0  # wrong prediction for target token 12
    corrupted, recovery, events = construct_self_prefix_batch(ids, labels, logits, horizon=2)
    assert events == 1
    assert corrupted.tolist() == [[10, 11, 9, 13, 14, 15]]
    assert recovery.tolist() == [[-100, -100, -100, 13, 14, -100]]


def test_construct_self_prefix_batch_skips_fully_correct_rows():
    ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[-100, 2, 3]])
    logits = torch.full((1, 3, 8), -10.0)
    logits[0, 0, 2] = 10.0
    logits[0, 1, 3] = 10.0
    corrupted, recovery, events = construct_self_prefix_batch(ids, labels, logits, horizon=16)
    assert events == 0
    assert torch.equal(corrupted, ids)
    assert bool((recovery == -100).all())
