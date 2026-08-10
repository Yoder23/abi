def test_amp_overflow_threshold_reasoning() -> None:
    assert 2.0 * 65536.0 > 65504.0
