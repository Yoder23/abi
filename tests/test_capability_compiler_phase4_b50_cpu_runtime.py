from abi.capability_compiler_phase4_b50_cpu_runtime import (
    FORMAT,
    RESULT_FORMAT,
    _ordinary_request,
    _paired_prompt_throughput,
    _paired_ratio_or_zero,
    _qwen_probe,
)


def test_cpu_runtime_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-cpu-runtime/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b50-cpu-runtime-result/1"


def test_qwen_probe_preserves_prompt_identity_and_limit():
    probe = {"probe_id": "p", "prompt": "hello", "max_new_tokens": 17}
    assert _qwen_probe(probe) == {
        "probe_id": "p",
        "prompt": "hello",
        "max_new_tokens": 17,
    }


def test_ordinary_request_is_callable_boundary():
    assert callable(_ordinary_request)


def test_paired_throughput_bootstrap_aggregates_repeats_by_prompt():
    candidate = [
        {"probe_id": "p0", "bytes_per_second": 10.0},
        {"probe_id": "p1", "bytes_per_second": 4.0},
        {"probe_id": "p0", "bytes_per_second": 30.0},
    ]
    baseline = [
        {"probe_id": "p0", "bytes_per_second": 5.0},
        {"probe_id": "p1", "bytes_per_second": 2.0},
        {"probe_id": "p0", "bytes_per_second": 15.0},
    ]
    left, right = _paired_prompt_throughput(candidate, baseline)
    assert left == [20.0, 4.0]
    assert right == [10.0, 2.0]


def test_cpu_zero_output_throughput_is_preserved_as_nonestimable():
    result = _paired_ratio_or_zero(
        [2.0, 1.0], [1.0, 0.0], replicates=100, seed=7
    )
    assert result["status"] == "NOT_ESTIMABLE_ZERO_OUTPUT_THROUGHPUT"
    assert result["zero_baseline_observations"] == 1
    assert result["lower_95"] is None
