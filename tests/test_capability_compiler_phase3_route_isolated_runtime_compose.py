import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_route_isolated_runtime_compose import compose


def _runtime():
    return {"device_control": {"candidate_fully_cpu": True, "candidate_cuda_allocated_before_bytes": 0, "candidate_cuda_allocated_after_bytes": 0, "candidate_cuda_peak_allocated_bytes": 0, "qwen_all_headline_size_vram_zero": True}, "optimized_transformer": {"digest": "d"}, "candidate": {"peak_active_rss_delta_bytes": 5}, "gates": {"quality": True, "lower_peak_active_rss": False}}


def _rss():
    return {"qwen": {"processor": "100% CPU", "size_vram_bytes": 0, "digest": "d", "monitored_peak_runner_working_set_bytes": 10}, "candidate": {"sealed_peak_process_rss_delta_bytes": 5}, "passed": True}


def test_compose_replaces_only_rss_gate():
    assert compose(_runtime(), _rss())["corrected_gates"] == {"quality": True, "lower_peak_active_rss": True}


def test_compose_rejects_candidate_binding_mismatch():
    rss = _rss()
    rss["candidate"]["sealed_peak_process_rss_delta_bytes"] = 6
    with pytest.raises(Phase3Error, match="differs"):
        compose(_runtime(), rss)
