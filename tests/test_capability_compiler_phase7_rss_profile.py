from abi.capability_compiler_phase7_rss_profile import _profile_stage


class _Memory:
    def __init__(self, owner):
        self.rss = owner.value


class _Process:
    def __init__(self):
        self.value = 100

    def memory_info(self):
        return _Memory(self)


def test_profile_stage_reports_delta_from_one_fixed_runtime_baseline():
    process = _Process()

    def operation():
        process.value = 175
        return "done"

    stage, value = _profile_stage(process, 100, "stage", operation)
    assert value == "done"
    assert stage["stage"] == "stage"
    assert stage["peak_delta_from_runtime_baseline_bytes"] == 75
