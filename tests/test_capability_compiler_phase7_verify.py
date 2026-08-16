import copy
import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase7_direct_artifact_runtime import load_protocol
from abi.capability_compiler_phase7_verify import _rows, verify_device_document


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_PROTOCOL = (
    ROOT / "ABI_CAPABILITY_COMPILER_PHASE7_ALLOCATION_BOUNDED_VERIFY_PROTOCOL_V1061.json"
)
RESULT = (
    ROOT
    / "results/abi_capability_compiler_phase7_integrated/allocation_bounded_verify_v1063/cpu/result.json"
)
OBSERVATIONS = RESULT.with_name("observations.jsonl")


@pytest.fixture(scope="module")
def exact_fixture():
    protocol, protocol_sha = load_protocol(ROOT, PRODUCT_PROTOCOL)
    return (
        protocol,
        protocol_sha,
        json.loads(RESULT.read_text(encoding="utf-8")),
        _rows(OBSERVATIONS),
    )


def _verify(fixture, mutation):
    protocol, protocol_sha, raw_result, raw_rows = fixture
    result = copy.deepcopy(raw_result)
    rows = copy.deepcopy(raw_rows)
    if mutation is not None:
        mutation(result, rows)
    return verify_device_document(
        root=ROOT,
        protocol=protocol,
        protocol_sha=protocol_sha,
        device="cpu",
        result=result,
        observations=rows,
    )


def test_exact_phase7_cpu_document_recomputes_cleanly(exact_fixture):
    assert all(_verify(exact_fixture, None).values())


@pytest.mark.parametrize(
    "mutation,failed_gate",
    [
        (lambda r, o: o.pop(), "raw_depth_partition_exact"),
        (
            lambda r, o: next(row for row in o if row["mode"] == "core_runtime").update(output="tampered"),
            "core_schedule_and_reference_exact",
        ),
        (
            lambda r, o: next(row for row in o if row["mode"] == "domain_runtime").update(output="repeat " * 100),
            "zero_repetition_collapse_recomputed",
        ),
        (
            lambda r, o: next(row for row in o if row["mode"] == "domain_runtime")["telemetry_delta"].update(evil={"prefill_calls": 1}),
            "domain_selected_only_recomputed",
        ),
        (
            lambda r, o: r["core_metrics"].update(median_bytes_per_second=0.0),
            "core_metrics_recomputed_exact",
        ),
        (
            lambda r, o: r["comparisons"].update(median_bytes_per_second_ratio_vs_baseline=1.0),
            "throughput_ratios_recomputed_exact",
        ),
        (
            lambda r, o: r["memory"].update(peak_process_rss_delta_bytes=10**12),
            "memory_gates_recomputed",
        ),
        (
            lambda r, o: r["core_after"].update(archive_hash="0" * 64),
            "core_identity_exact_and_unchanged",
        ),
        (
            lambda r, o: r["package_installs"]["python"].update(archive_hash="0" * 64),
            "package_identity_exact",
        ),
        (
            lambda r, o: r.update(serving_lifecycle="wrong"),
            "lifecycle_repair_exact",
        ),
        (
            lambda r, o: r.update(teacher_model_loaded=True),
            "teacher_training_receiver_learning_absent",
        ),
        (
            lambda r, o: r["gates"].update(evil=False),
            "declared_gates_all_pass",
        ),
        (
            lambda r, o: next(row for row in o if row["mode"] == "single_cold_core_request").update(time_to_first_output_from_cold_start_seconds=10**6),
            "cold_single_request_and_ttft_recomputed",
        ),
        (
            lambda r, o: next(row for row in o if row["mode"] == "core_runtime" and row["capability"] == "grammar")["execution"].update(persistent_state_created=False),
            "persistent_incremental_state_recomputed",
        ),
        (
            lambda r, o: r.update(status="FAIL_PHASE7_INTEGRATED_RUNTIME"),
            "result_format_status_device_protocol",
        ),
    ],
)
def test_adversarial_phase7_mutations_fail_closed(
    exact_fixture, mutation, failed_gate
):
    assert _verify(exact_fixture, mutation)[failed_gate] is False
