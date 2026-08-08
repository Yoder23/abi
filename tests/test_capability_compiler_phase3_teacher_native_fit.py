from abi.capability_compiler_phase3_teacher_native_fit import summarize_counts


def test_summarize_counts_is_exact():
    value = summarize_counts(10, 9, 2, 1, 1.0, 3)
    assert value["action_accuracy"] == 0.9
    assert value["exact_sequence_rate"] == 0.5
    assert value["mean_nll"] == 0.1
    assert value["pointer_argmax_actions"] == 3
