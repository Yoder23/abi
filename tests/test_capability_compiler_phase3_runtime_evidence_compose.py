from abi.capability_compiler_phase3_runtime_evidence_compose import compose


def test_compose_replaces_only_rss_gate():
    runtime = {"candidate_device": {"fully_cpu": True, "cuda_allocated_before_bytes": 0, "cuda_allocated_after_bytes": 0, "cuda_peak_allocated_bytes": 0}, "optimized_transformer": {"digest": "d"}, "candidate": {"peak_active_rss_delta_bytes": 5}, "gates": {"quality": True, "lower_peak_active_rss": False}}
    rss = {"qwen": {"processor": "100% CPU", "size_vram_bytes": 0, "digest": "d", "monitored_peak_runner_working_set_bytes": 10}}
    assert compose(runtime, rss)["corrected_gates"] == {"quality": True, "lower_peak_active_rss": True}
