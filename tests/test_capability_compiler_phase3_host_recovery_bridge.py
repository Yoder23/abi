from abi.capability_compiler_phase3_host_recovery_bridge import AllStrataSampler


def test_all_strata_sampler_covers_each_capability_builder_once_per_batch():
    capabilities = ("abstention", "coherence", "fluent_realization", "tone_control")
    rows = [
        {"record_id": f"{capability}-{builder}-{index}", "capability": capability, "builder": builder}
        for capability in capabilities
        for builder in range(4)
        for index in range(80)
    ]
    sampler = AllStrataSampler(rows, 7)
    batch = sampler.teacher_forced_batch()
    assert len(batch) == 16
    assert {(row["capability"], row["builder"]) for row in batch} == {
        (capability, builder) for capability in capabilities for builder in range(4)
    }


def test_recovery_sampler_cycles_all_strata_without_omission():
    capabilities = ("abstention", "coherence", "fluent_realization", "tone_control")
    rows = [
        {"record_id": f"{capability}-{builder}-{index}", "capability": capability, "builder": builder}
        for capability in capabilities
        for builder in range(4)
        for index in range(80)
    ]
    sampler = AllStrataSampler(rows, 9)
    selected = [*sampler.recovery_batch(4), *sampler.recovery_batch(4), *sampler.recovery_batch(4), *sampler.recovery_batch(4)]
    assert {(row["capability"], row["builder"]) for row in selected} == {
        (capability, builder) for capability in capabilities for builder in range(4)
    }
