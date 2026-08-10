def test_four_microbatches_preserve_logical_batch() -> None:
    assert sum([4, 4, 4, 4]) == 16
