def test_fit_attribution_thresholds_are_conjunctive():
    action_accuracy = 0.995
    exact_sequence_rate = 0.85
    assert action_accuracy >= 0.99
    assert not exact_sequence_rate >= 0.90
    assert not (action_accuracy >= 0.99 and exact_sequence_rate >= 0.90)
