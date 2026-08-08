def test_native_causal_payload_accounting():
    assert 208647 * 192 * 2 == 80_120_448
    assert (7000 + 1) * 8 == 56_008
