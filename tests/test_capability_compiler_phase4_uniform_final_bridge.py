from pathlib import Path

from abi.capability_compiler_phase4_uniform_final_bridge import UniformDualViewSampler


def _rows(depth: int):
    result = []
    for capability in ("fluent_realization", "coherence", "tone_control", "abstention"):
        for builder in range(4):
            for view in ("host_projected", "source_wrapped"):
                for index in range(depth):
                    result.append({"record_id": f"{capability}-{builder}-{view}-{index:03d}", "capability": capability, "builder": builder, "view": view})
    return result


def test_uniform_sampler_balances_records_and_recovery_reuses_current_row() -> None:
    sampler = UniformDualViewSampler(_rows(64), seed=104729)
    for step in range(2000):
        selected = sampler.teacher_forced_batch()
        if step >= 99 and (step - 99) % 4 == 0:
            recovered = sampler.recovery_batch(8)
            by_key = {(row["capability"], row["builder"], row["view"]): row["record_id"] for row in selected}
            assert all(by_key[(row["capability"], row["builder"], row["view"])] == row["record_id"] for row in recovered)
    profile = sampler.profile()
    assert profile["seed_dependent_sampling"] is False
    assert profile["maximum_within_stratum_exposure_range"] == 1
    assert {row["minimum_exposures"] for row in profile["strata"].values()} == {31}
    assert {row["maximum_exposures"] for row in profile["strata"].values()} == {32}


def test_uniform_sampler_sequence_is_seed_invariant() -> None:
    left = UniformDualViewSampler(_rows(32), seed=1)
    right = UniformDualViewSampler(_rows(32), seed=999)
    for _ in range(65):
        assert [row["record_id"] for row in left.teacher_forced_batch()] == [row["record_id"] for row in right.teacher_forced_batch()]
