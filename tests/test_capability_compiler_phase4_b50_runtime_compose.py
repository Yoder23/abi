from abi.capability_compiler_phase4_b50_runtime_compose import (
    FORMAT,
    RESULT_FORMAT,
    _identity,
    _metrics_equal,
    _paired_or_zero,
)


def test_runtime_compose_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-runtime-compose/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b50-runtime-compose-result/1"


def test_cpu_gpu_identity_requires_same_prompt_output_and_tokens():
    left = [{"probe_id": "p", "output": "ok", "output_token_ids": [1]}]
    assert _identity(left, left) == 1
    assert _identity(left, [{**left[0], "output_token_ids": [2]}]) == 0


def test_metrics_equal_uses_primary_cross_model_metrics():
    value = {
        "observations": 120,
        "median_bytes_per_second": 2.0,
        "median_characters_per_second": 1.0,
        "median_time_to_first_output_seconds": 0.1,
        "median_total_seconds": 0.2,
        "p95_supported": True,
        "p95_time_to_first_output_seconds": 0.15,
        "p95_total_seconds": 0.3,
        "p05_bytes_per_second": 1.0,
        "p05_characters_per_second": 0.5,
        "p99_supported": False,
        "p99_time_to_first_output_seconds": None,
        "p99_total_seconds": None,
    }
    assert _metrics_equal(value, dict(value))
    assert not _metrics_equal(value, {**value, "median_bytes_per_second": 3.0})


def test_zero_output_throughput_is_reported_not_divided_away():
    result = _paired_or_zero([2.0, 1.0], [1.0, 0.0], replicates=100, seed=7)
    assert result["status"] == "NOT_ESTIMABLE_ZERO_OUTPUT_THROUGHPUT"
    assert result["zero_baseline_observations"] == 1
    assert result["lower_95"] is None
