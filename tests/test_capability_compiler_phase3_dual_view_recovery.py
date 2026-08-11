from abi.capability_compiler_phase3_dual_view_recovery import DualViewSampler


def _rows():
    return [
        {"record_id": f"{capability}-{builder}-{view}-{index}", "capability": capability, "builder": builder, "view": view}
        for capability in ("abstention", "coherence", "fluent_realization", "tone_control")
        for builder in range(4)
        for view in ("host_projected", "source_wrapped")
        for index in range(80)
    ]


def test_teacher_batches_cover_all_32_view_strata():
    sampler = DualViewSampler(_rows(), 7)
    batch = sampler.teacher_forced_batch()
    assert len(batch) == 32
    assert {row["view"] for row in batch} == {"host_projected", "source_wrapped"}
    assert len({(row["capability"], row["builder"], row["view"]) for row in batch}) == 32


def test_four_recovery_batches_cover_all_32_view_strata():
    sampler = DualViewSampler(_rows(), 9)
    selected = [row for _ in range(4) for row in sampler.recovery_batch(8)]
    assert len({(row["capability"], row["builder"], row["view"]) for row in selected}) == 32
