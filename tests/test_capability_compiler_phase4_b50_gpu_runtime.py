from abi.capability_compiler_phase4_b50_gpu_runtime import (
    FORMAT,
    RESULT_FORMAT,
    SYSTEMS,
    _identity,
    _observation,
    _round_robin,
    _runtime_metrics,
)


def test_gpu_runtime_contract_is_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-gpu-runtime/1"
    assert RESULT_FORMAT == "abi-capability-compiler-phase4-b50-gpu-runtime-result/1"
    assert SYSTEMS == ("ABI", "L0", "L1", "D0", "D1", "D2")


def test_observation_uses_bytes_characters_and_completed_response_tokens():
    row = _observation(
        probe={"probe_id": "p", "canonical_capability": "grammar"},
        output="café",
        output_token_ids=[1, 2],
        first_seconds=0.1,
        total_seconds=0.5,
    )
    assert row["output_utf8_bytes"] == 5
    assert row["output_characters"] == 4
    assert row["authoritative_output_tokens"] == 2
    assert row["bytes_per_second"] == 10
    assert row["token_accounting"] == "completed_response_retokenization"


def test_observation_separates_generation_identity_from_retokenized_accounting():
    row = _observation(
        probe={"probe_id": "p", "canonical_capability": "grammar"},
        output="joined token surface",
        output_token_ids=[10, 11, 12],
        retokenized_output_token_ids=[20, 21],
        first_seconds=0.1,
        total_seconds=0.5,
    )
    assert row["output_token_ids"] == [10, 11, 12]
    assert row["retokenized_output_token_ids"] == [20, 21]
    assert row["authoritative_output_tokens"] == 2


def test_identity_requires_output_and_token_ids():
    reference = {"p": {"output": "ok", "output_token_ids": [1, 2]}}
    row = {"probe_id": "p", "output": "ok", "output_token_ids": [1, 2]}
    assert _identity([row], reference) == 1
    assert _identity([{**row, "output_token_ids": [2]}], reference) == 0


def test_runtime_metrics_promote_p95_only_at_supported_depth():
    template = {
        "bytes_per_second": 10.0,
        "characters_per_second": 8.0,
        "time_to_first_output_seconds": 0.1,
        "total_seconds": 0.5,
    }
    shallow = _runtime_metrics([template] * 20)
    assert shallow["p95_supported"] is False
    assert shallow["p95_total_seconds"] is None
    deep = _runtime_metrics([template] * 100)
    assert deep["p95_supported"] is True
    assert deep["p95_total_seconds"] == 0.5
    assert deep["p99_supported"] is False
    assert deep["p99_total_seconds"] is None


def test_runtime_schedule_helper_interleaves_capabilities():
    groups = [
        [{"probe_id": "a0"}, {"probe_id": "a1"}],
        [{"probe_id": "b0"}],
        [{"probe_id": "c0"}, {"probe_id": "c1"}],
    ]
    assert [row["probe_id"] for row in _round_robin(groups)] == [
        "a0",
        "b0",
        "c0",
        "a1",
        "c1",
    ]
