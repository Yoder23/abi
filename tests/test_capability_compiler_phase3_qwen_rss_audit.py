from abi.capability_compiler_phase3_qwen_rss_audit import runner_working_set


def test_runner_working_set_is_nonnegative():
    assert runner_working_set() >= 0
