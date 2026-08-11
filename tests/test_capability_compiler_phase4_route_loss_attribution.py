from abi import capability_compiler_phase4_route_loss_attribution as subject


def test_partition_is_deterministic_and_nontrivial() -> None:
    values = [subject._partition(f"record-{index}", 5) for index in range(100)]
    assert values == [subject._partition(f"record-{index}", 5) for index in range(100)]
    assert set(values) == set(range(5))


def test_per_record_loss_ignores_prompt_labels() -> None:
    import torch
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[-100, -100, 1, 2]])
    value = subject._per_record_loss(logits, labels)
    assert value.shape == (1,)
    assert torch.isfinite(value).all()
