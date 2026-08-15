"""Independently compose exact-B50 CPU/GPU same-artifact runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_cpu_runtime import paired_ratio_bootstrap
from .capability_compiler_phase4_b50_cpu_runtime import (
    RESULT_FORMAT as CPU_RESULT_FORMAT,
    _paired_prompt_throughput,
    _paired_ratio_or_zero,
)
from .capability_compiler_phase4_b50_gpu_runtime import (
    RESULT_FORMAT as GPU_RESULT_FORMAT,
    SYSTEMS,
    _runtime_metrics,
    runtime_schedule,
)
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase4_v19_cpu_runtime import paired_quality_bootstrap
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-runtime-compose/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-runtime-compose-result/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_INDEPENDENT_B50_RUNTIME_COMPOSITION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("matched B50 runtime composition governance changed")
    if set(protocol.get("gpu_results", {})) != set(SYSTEMS):
        raise Phase3Error("matched B50 runtime composition matrix changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"matched B50 runtime composition binding changed: {relative}")
    return protocol, sha256_file(path)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def _metrics_equal(recorded: Mapping[str, Any], recomputed: Mapping[str, Any]) -> bool:
    return all(
        recorded[key] == recomputed[key]
        for key in (
            "observations",
            "median_bytes_per_second",
            "median_characters_per_second",
            "median_time_to_first_output_seconds",
            "median_total_seconds",
            "p95_supported",
            "p95_time_to_first_output_seconds",
            "p95_total_seconds",
            "p05_bytes_per_second",
            "p05_characters_per_second",
            "p99_supported",
            "p99_time_to_first_output_seconds",
            "p99_total_seconds",
        )
    )


def _identity(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    if [row["probe_id"] for row in left] != [row["probe_id"] for row in right]:
        raise Phase3Error("matched B50 CPU/GPU prompt order changed")
    return sum(
        row_left["output"] == row_right["output"]
        and row_left["output_token_ids"] == row_right["output_token_ids"]
        for row_left, row_right in zip(left, right)
    )


def _paired_or_zero(
    candidate: list[float],
    baseline: list[float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    zero_candidate = sum(value <= 0 for value in candidate)
    zero_baseline = sum(value <= 0 for value in baseline)
    if zero_candidate or zero_baseline:
        return {
            "status": "NOT_ESTIMABLE_ZERO_OUTPUT_THROUGHPUT",
            "observations": len(candidate),
            "zero_candidate_observations": zero_candidate,
            "zero_baseline_observations": zero_baseline,
            "replicates": 0,
            "seed": seed,
            "lower_95": None,
            "upper_95": None,
        }
    return {
        "status": "ESTIMABLE",
        **paired_ratio_bootstrap(candidate, baseline, replicates, seed),
        "zero_candidate_observations": 0,
        "zero_baseline_observations": 0,
    }


def _ratio(left: float, right: float) -> float | None:
    return left / right if right > 0 else None


def _recompute_cpu_evidence(
    root: Path,
    runtime_protocol: Mapping[str, Any],
    cpu: Mapping[str, Any],
    product_rows: list[dict[str, Any]],
    ordinary_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    distinct, scheduled = runtime_schedule(root, runtime_protocol)
    expected_order = [str(row["probe_id"]) for row in scheduled]
    if any(
        [str(row["probe_id"]) for row in rows] != expected_order
        for rows in (product_rows, ordinary_rows, qwen_rows)
    ):
        raise Phase3Error("matched B50 CPU prompt order changed")
    candidate_spec = runtime_protocol["systems"]["ABI"]
    reference_path = root / str(candidate_spec["quality_reference_outputs"])
    if sha256_file(reference_path) != candidate_spec["quality_reference_sha256"]:
        raise Phase3Error("matched B50 CPU quality reference changed")
    reference = {
        str(row["probe_id"]): row for row in _jsonl(reference_path)
    }
    identities = sum(
        str(row["output"]) == str(reference[str(row["probe_id"])]["output"])
        and [int(value) for value in row["output_token_ids"]]
        == [
            int(value)
            for value in reference[str(row["probe_id"])]["output_token_ids"]
        ]
        for row in product_rows
    )
    prompt_candidate, prompt_qwen = _paired_prompt_throughput(
        product_rows, qwen_rows
    )
    paired_speed = _paired_ratio_or_zero(
        prompt_candidate,
        prompt_qwen,
        replicates=int(runtime_protocol["statistics"]["bootstrap_replicates"]),
        seed=int(runtime_protocol["statistics"]["throughput_bootstrap_seed"]),
    )
    paired_speed["method"] = (
        "paired_prompt_median_throughput_ratio_percentile_bootstrap"
    )
    paired_speed["prompt_pairs"] = len(prompt_candidate)
    candidate_v1 = [
        evaluate_functional(str(row["output"]), probe["evaluator"])
        for row, probe in zip(product_rows[: len(distinct)], distinct)
    ]
    candidate_v2 = [
        evaluate_functional_v2(
            str(row["output"]),
            probe["evaluator"],
            str(probe["canonical_capability"]),
        )
        for row, probe in zip(product_rows[: len(distinct)], distinct)
    ]
    qwen_v1 = [
        evaluate_functional(str(row["output"]), probe["evaluator"])
        for row, probe in zip(qwen_rows[: len(distinct)], distinct)
    ]
    qwen_v2 = [
        evaluate_functional_v2(
            str(row["output"]),
            probe["evaluator"],
            str(probe["canonical_capability"]),
        )
        for row, probe in zip(qwen_rows[: len(distinct)], distinct)
    ]
    paired_quality_v1 = paired_quality_bootstrap(
        candidate_v1,
        qwen_v1,
        int(runtime_protocol["statistics"]["bootstrap_replicates"]),
        int(runtime_protocol["statistics"]["quality_bootstrap_seed"]),
    )
    paired_quality_v2 = paired_quality_bootstrap(
        candidate_v2,
        qwen_v2,
        int(runtime_protocol["statistics"]["bootstrap_replicates"]),
        int(runtime_protocol["statistics"]["quality_bootstrap_seed"]) + 1,
    )
    candidate_metrics = _runtime_metrics(product_rows)
    ordinary_metrics = _runtime_metrics(ordinary_rows)
    qwen_metrics = _runtime_metrics(qwen_rows)
    throughput_ratio = (
        candidate_metrics["median_bytes_per_second"]
        / qwen_metrics["median_bytes_per_second"]
    )
    ttft_ratio = (
        candidate_metrics["median_time_to_first_output_seconds"]
        / qwen_metrics["median_time_to_first_output_seconds"]
    )
    retention = ordinary_metrics["median_bytes_per_second"] / float(
        runtime_protocol["locked_phase2_runtime"]["median_bytes_per_second"]
    )
    candidate = cpu["candidate"]
    qwen = cpu["optimized_transformer"]
    comparisons = {
        "median_throughput_ratio": throughput_ratio,
        "paired_throughput": paired_speed,
        "paired_quality_v1": paired_quality_v1,
        "paired_quality_v2": paired_quality_v2,
        "median_ttft_ratio": ttft_ratio,
        "cold_total_latency_ratio": candidate["cold"][
            "total_from_cold_start_seconds"
        ]
        / qwen["cold"]["total_seconds"],
        "peak_rss_ratio": candidate["peak_active_rss_delta_bytes"]
        / qwen["peak_runner_rss_bytes"],
        "ordinary_phase2_throughput_retention": retention,
    }
    pointer_rows = [row for row in product_rows if row["capability"] == "coherence"]
    format_rows = [
        row for row in product_rows if row["capability"] == "format_control"
    ]
    ordinary_product_rows = [
        row
        for row in product_rows
        if row["capability"] not in {"coherence", "format_control"}
    ]
    device = cpu["device_control"]
    activation = cpu["package"]
    gates_cfg = runtime_protocol["gates"]
    gates = {
        "same_signed_package": activation["archive_sha256"]
        == candidate_spec["archive_sha256"],
        "payload_preserved": activation["tensor_payload_hash"]
        == candidate_spec["tensor_payload_hash"],
        "runtime_outputs_exact_to_quality_candidate": identities
        == len(product_rows),
        "quality_v1_noninferior_to_qwen": paired_quality_v1["lower_95"]
        >= float(gates_cfg["quality_relative_lower_minimum"]),
        "quality_v2_noninferior_to_qwen": paired_quality_v2["lower_95"]
        >= float(gates_cfg["quality_relative_lower_minimum"]),
        "throughput_ratio_at_least_2x": throughput_ratio
        >= float(gates_cfg["cpu_throughput_ratio_minimum"]),
        "paired_throughput_lower_at_least_2x": paired_speed["lower_95"]
        is not None
        and paired_speed["lower_95"]
        >= float(gates_cfg["paired_bootstrap_lower_minimum"]),
        "ordinary_phase2_throughput_retention": retention
        >= float(gates_cfg["phase2_host_throughput_retention_minimum"]),
        "ttft_advantage": ttft_ratio <= float(gates_cfg["ttft_ratio_maximum"]),
        "cold_ttft_no_worse": candidate["cold"][
            "time_to_first_output_from_cold_start_seconds"
        ]
        <= qwen["cold"]["time_to_first_output_seconds"],
        "lower_active_tensor_bytes": candidate["active_tensor_bytes"]
        < int(runtime_protocol["transformer_baseline"]["model_file_bytes"]),
        "lower_peak_active_rss": candidate["peak_active_rss_delta_bytes"]
        < qwen["peak_runner_rss_bytes"],
        "candidate_fully_cpu": device["candidate_cuda_allocated_before_bytes"]
        == device["candidate_cuda_allocated_after_bytes"]
        == device["candidate_cuda_peak_allocated_bytes"]
        == 0,
        "qwen_fully_cpu": device["qwen_device_observations"] == 121
        and device["qwen_size_vram_zero_observations"] == 121,
        "genuine_candidate_cold_single_request": candidate["cold"][
            "single_cold_request"
        ]
        and candidate["cold"]["model_load_seconds"] > 0,
        "genuine_qwen_cold_single_request": qwen["cold"]["single_cold_request"]
        and qwen["cold"]["load_seconds_reported"] > 0,
        "authoritative_token_accounting": all(
            row["token_accounting"] == "completed_response_retokenization"
            and row["authoritative_output_tokens"]
            == len(row["retokenized_output_token_ids"])
            for row in product_rows + ordinary_rows
        )
        and all(
            row["token_accounting"] == "authoritative_runtime_eval_count"
            and row["authoritative_output_tokens"] >= 0
            for row in qwen_rows
        ),
        "depth": len(product_rows) == len(qwen_rows) == 120
        and len({row["probe_id"] for row in product_rows}) == 100
        and paired_speed["prompt_pairs"] == 100,
        "candidate_zero_repetition_collapse_v2": not any(
            repetition_collapse_v2(str(row["output"])) for row in product_rows
        ),
        "p95_supported": candidate_metrics["p95_supported"]
        and qwen_metrics["p95_supported"],
        "p99_not_promoted": not candidate_metrics["p99_supported"]
        and not qwen_metrics["p99_supported"],
        "receiver_learning_zero": cpu["receiver_training_steps"]
        == cpu["receiver_calibration_runs"]
        == 0,
        "pointer_physical_execution": bool(pointer_rows)
        and all(
            row["execution"]["pointer"].get("candidate_count") == 6
            and row["execution"]["pointer"].get("candidate_scoring_forward_passes")
            == 1
            and row["execution"]["pointer"].get("active_residual_routes") == 1
            and row["execution"]["pointer"].get("persistent_prompt_state_reused")
            is True
            and row["execution"]["pointer"].get("evaluator_used") is False
            for row in pointer_rows
        ),
        "format_physical_execution": bool(format_rows)
        and all(
            row["execution"]["format"].get("deterministic_transducer") is True
            and row["execution"]["format"].get("prompt_prefill_forward_passes") == 1
            and row["execution"]["format"].get("candidate_scoring_forward_passes")
            == 0
            and row["execution"]["format"].get("decode_forward_passes") == 0
            and row["execution"]["format"].get("active_residual_routes") == 0
            for row in format_rows
        ),
        "ordinary_persistent_state_and_route": bool(ordinary_product_rows)
        and all(
            row["execution"].get("persistent_state_created") is True
            and row["execution"].get("route_correct") is True
            for row in ordinary_product_rows
        ),
        "teacher_absent": cpu.get("teacher_present") is False,
        "training_absent": cpu.get("training_performed") is False,
        "final_test_not_accessed": cpu.get("final_test_accessed") is False,
    }
    return gates, comparisons


def _gpu_result(
    root: Path,
    path: Path,
    *,
    expected_system: str,
    expected_protocol_sha256: str,
    runtime_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _json(path)
    if (
        result.get("format") != GPU_RESULT_FORMAT
        or result.get("system") != expected_system
        or result.get("protocol_sha256") != expected_protocol_sha256
        or not result_evidence_digest_valid(result)
        or result.get("training_performed") is not False
        or result.get("final_test_accessed") is not False
    ):
        raise Phase3Error(f"matched B50 GPU result changed: {expected_system}")
    observations = root / str(result["observations_path"])
    if sha256_file(observations) != result["observations_sha256"]:
        raise Phase3Error(f"matched B50 GPU observations changed: {expected_system}")
    rows = _jsonl(observations)
    if len(rows) != 120 or not _metrics_equal(result["metrics"], _runtime_metrics(rows)):
        raise Phase3Error(f"matched B50 GPU metrics changed: {expected_system}")
    reference_path = root / str(
        runtime_protocol["systems"][expected_system]["quality_reference_outputs"]
    )
    if (
        sha256_file(reference_path)
        != runtime_protocol["systems"][expected_system]["quality_reference_sha256"]
    ):
        raise Phase3Error(f"matched B50 GPU quality reference changed: {expected_system}")
    reference = {
        str(row["probe_id"]): row for row in _jsonl(reference_path)
    }
    output_identities = sum(
        str(row["output"]) == str(reference[str(row["probe_id"])]["output"])
        and [int(value) for value in row["output_token_ids"]]
        == [
            int(value)
            for value in reference[str(row["probe_id"])]["output_token_ids"]
        ]
        for row in rows
    )
    metrics = _runtime_metrics(rows)
    physical_execution = True
    route_execution = True
    if expected_system == "ABI":
        pointer_rows = [row for row in rows if row["capability"] == "coherence"]
        format_rows = [row for row in rows if row["capability"] == "format_control"]
        ordinary_rows = [
            row
            for row in rows
            if row["capability"] not in {"coherence", "format_control"}
        ]
        physical_execution = bool(pointer_rows) and bool(format_rows) and all(
            row["execution"]["pointer"].get("candidate_count") == 6
            and row["execution"]["pointer"].get("candidate_scoring_forward_passes")
            == 1
            and row["execution"]["pointer"].get("active_residual_routes") == 1
            and row["execution"]["pointer"].get("persistent_prompt_state_reused")
            is True
            and row["execution"]["pointer"].get("evaluator_used") is False
            for row in pointer_rows
        ) and all(
            row["execution"]["format"].get("deterministic_transducer") is True
            and row["execution"]["format"].get("prompt_prefill_forward_passes") == 1
            and row["execution"]["format"].get("candidate_scoring_forward_passes")
            == 0
            and row["execution"]["format"].get("decode_forward_passes") == 0
            and row["execution"]["format"].get("active_residual_routes") == 0
            for row in format_rows
        )
        route_execution = bool(ordinary_rows) and all(
            row["execution"].get("route_correct") is True for row in ordinary_rows
        )
    elif expected_system in {"L0", "L1"}:
        route_execution = all(
            row["execution"].get("route_correct") is True for row in rows
        )
    recomputed_gates = {
        "quality_output_identity": output_identities == len(rows),
        "depth": len(rows) == 120
        and len({str(row["probe_id"]) for row in rows}) == 100,
        "authoritative_token_accounting": all(
            row.get("token_accounting") == "completed_response_retokenization"
            and int(row["authoritative_output_tokens"])
            == len(row["retokenized_output_token_ids"])
            for row in rows
        ),
        "p95_supported": bool(metrics["p95_supported"]),
        "p99_not_promoted": not bool(metrics["p99_supported"]),
        "single_cold_request": result["cold"].get("single_cold_request") is True
        and float(result["cold"].get("model_load_seconds", 0.0)) > 0,
        "physical_sparse_execution": physical_execution,
        "route_execution": route_execution,
        "teacher_query_absent": result.get("teacher_query_performed") is False,
        "training_absent": result.get("training_performed") is False,
        "final_test_not_accessed": result.get("final_test_accessed") is False,
    }
    expected_status = (
        "PASS_SAME_CHECKPOINT_B50_CUDA_RUNTIME"
        if all(recomputed_gates.values())
        else "FAIL_SAME_CHECKPOINT_B50_CUDA_RUNTIME"
    )
    if (
        result.get("gates") != recomputed_gates
        or result.get("status") != expected_status
        or int(result.get("quality_output_identities", -1)) != output_identities
    ):
        raise Phase3Error(f"matched B50 GPU gates changed: {expected_system}")
    return result, rows


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    gpu_protocol_path = root / str(protocol["gpu_protocol"])
    gpu_protocol_sha = sha256_file(gpu_protocol_path)
    gpu_runtime_protocol = _json(gpu_protocol_path)
    cpu_protocol_path = root / str(protocol["cpu_protocol"])
    cpu_protocol_sha = sha256_file(cpu_protocol_path)
    cpu_runtime_protocol = _json(cpu_protocol_path)
    quality = _json(root / str(protocol["matched_quality_result"]))
    if (
        quality.get("status")
        != "PASS_READ_ONLY_MATCHED_B50_QUALITY_AND_COST_RECOMPUTED"
        or not result_evidence_digest_valid(quality)
    ):
        raise Phase3Error("matched B50 runtime quality prerequisite changed")
    gpu_results = {}
    gpu_rows = {}
    for system in SYSTEMS:
        result, rows = _gpu_result(
            root,
            root / str(protocol["gpu_results"][system]),
            expected_system=system,
            expected_protocol_sha256=gpu_protocol_sha,
            runtime_protocol=gpu_runtime_protocol,
        )
        gpu_results[system] = result
        gpu_rows[system] = rows

    cpu = _json(root / str(protocol["cpu_result"]))
    if (
        cpu.get("format") != CPU_RESULT_FORMAT
        or cpu.get("protocol_sha256") != cpu_protocol_sha
        or not result_evidence_digest_valid(cpu)
        or cpu.get("training_performed") is not False
        or cpu.get("final_test_accessed") is not False
    ):
        raise Phase3Error("matched B50 CPU result changed")
    cpu_observations_path = root / str(protocol["cpu_observations"])
    if sha256_file(cpu_observations_path) != cpu["raw_observations_sha256"]:
        raise Phase3Error("matched B50 CPU observations changed")
    cpu_all = _jsonl(cpu_observations_path)
    cpu_product = [
        row
        for row in cpu_all
        if row.get("system") == "layercake_v22_b50" and row.get("mode") == "product"
    ]
    cpu_ordinary = [
        row
        for row in cpu_all
        if row.get("system") == "layercake_v22_b50" and row.get("mode") == "ordinary"
    ]
    cpu_qwen = [row for row in cpu_all if row.get("system") == "qwen"]
    if not all(len(rows) == 120 for rows in (cpu_product, cpu_ordinary, cpu_qwen)):
        raise Phase3Error("matched B50 CPU observation depth changed")
    if (
        not _metrics_equal(cpu["candidate"], _runtime_metrics(cpu_product))
        or not _metrics_equal(
            cpu["candidate"]["ordinary"], _runtime_metrics(cpu_ordinary)
        )
        or not _metrics_equal(
            cpu["optimized_transformer"], _runtime_metrics(cpu_qwen)
        )
    ):
        raise Phase3Error("matched B50 CPU aggregate changed")
    cpu_gates, cpu_comparisons = _recompute_cpu_evidence(
        root,
        cpu_runtime_protocol,
        cpu,
        cpu_product,
        cpu_ordinary,
        cpu_qwen,
    )
    expected_cpu_status = (
        "PASS_SAME_ARTIFACT_B50_V22_CPU_RUNTIME"
        if all(cpu_gates.values())
        else "FAIL_SAME_ARTIFACT_B50_V22_CPU_RUNTIME"
    )
    if (
        cpu.get("gates") != cpu_gates
        or cpu.get("comparisons") != cpu_comparisons
        or cpu.get("status") != expected_cpu_status
    ):
        raise Phase3Error("matched B50 CPU gates or comparisons changed")

    gpu_candidate = gpu_results["ABI"]
    comparisons = {}
    quality_qualified = []
    for index, system in enumerate(SYSTEMS[1:]):
        baseline = gpu_results[system]
        prompt_candidate_bytes, prompt_baseline_bytes = _paired_prompt_throughput(
            gpu_rows["ABI"], gpu_rows[system]
        )
        bytes_bootstrap = _paired_or_zero(
            prompt_candidate_bytes,
            prompt_baseline_bytes,
            replicates=int(protocol["statistics"]["bootstrap_replicates"]),
            seed=int(protocol["statistics"]["bytes_seed_base"]) + index,
        )
        prompt_candidate_characters, prompt_baseline_characters = (
            _paired_prompt_throughput(
                [
                    {**row, "bytes_per_second": row["characters_per_second"]}
                    for row in gpu_rows["ABI"]
                ],
                [
                    {**row, "bytes_per_second": row["characters_per_second"]}
                    for row in gpu_rows[system]
                ],
            )
        )
        characters_bootstrap = _paired_or_zero(
            prompt_candidate_characters,
            prompt_baseline_characters,
            replicates=int(protocol["statistics"]["bootstrap_replicates"]),
            seed=int(protocol["statistics"]["characters_seed_base"]) + index,
        )
        for bootstrap, unit in (
            (bytes_bootstrap, "bytes"),
            (characters_bootstrap, "characters"),
        ):
            bootstrap["method"] = (
                f"paired_prompt_median_{unit}_throughput_ratio_percentile_bootstrap"
            )
            bootstrap["prompt_pairs"] = len(prompt_candidate_bytes)
        decision = quality["system_decisions"][system]
        qualified = bool(decision["baseline_all_seed_quality_pass"])
        if qualified:
            quality_qualified.append(system)
        ratios = {
            "median_bytes_per_second": _ratio(
                gpu_candidate["metrics"]["median_bytes_per_second"],
                baseline["metrics"]["median_bytes_per_second"],
            ),
            "median_characters_per_second": _ratio(
                gpu_candidate["metrics"]["median_characters_per_second"],
                baseline["metrics"]["median_characters_per_second"],
            ),
            "median_ttft": _ratio(
                gpu_candidate["metrics"]["median_time_to_first_output_seconds"],
                baseline["metrics"]["median_time_to_first_output_seconds"],
            ),
            "active_tensor_bytes": _ratio(
                gpu_candidate["active_tensor_bytes"], baseline["active_tensor_bytes"]
            ),
            "peak_process_rss": _ratio(
                gpu_candidate["peak_process_rss_delta_bytes"],
                baseline["peak_process_rss_delta_bytes"],
            ),
            "peak_cuda": _ratio(
                gpu_candidate["peak_cuda_allocated_bytes"],
                baseline["peak_cuda_allocated_bytes"],
            ),
            "cold_ttft": _ratio(
                gpu_candidate["cold"]["time_to_first_output_from_cold_start_seconds"],
                baseline["cold"]["time_to_first_output_from_cold_start_seconds"],
            ),
        }
        gates = {
            "both_runtime_results_pass": gpu_candidate["status"].startswith("PASS")
            and baseline["status"].startswith("PASS"),
            "candidate_quality_noninferior": bool(
                decision["candidate_noninferior_all_seed"]
            ),
            "median_gpu_throughput_not_slower": ratios["median_bytes_per_second"]
            is not None
            and ratios["median_bytes_per_second"]
            >= float(protocol["gates"]["quality_qualified_gpu_throughput_ratio_minimum"]),
            "paired_gpu_throughput_not_slower": bytes_bootstrap["lower_95"]
            is not None
            and bytes_bootstrap["lower_95"]
            >= float(protocol["gates"]["quality_qualified_gpu_paired_lower_minimum"]),
            "gpu_ttft_not_slower": ratios["median_ttft"] is not None
            and ratios["median_ttft"]
            <= float(protocol["gates"]["quality_qualified_gpu_ttft_ratio_maximum"]),
            "gpu_active_tensor_bytes_lower": ratios["active_tensor_bytes"] is not None
            and ratios["active_tensor_bytes"] < 1.0,
            "gpu_peak_rss_lower": ratios["peak_process_rss"] is not None
            and ratios["peak_process_rss"] < 1.0,
            "gpu_peak_cuda_lower": ratios["peak_cuda"] is not None
            and ratios["peak_cuda"] < 1.0,
        }
        comparisons[system] = {
            "baseline_quality_qualified": qualified,
            "ratios": ratios,
            "paired_bytes_per_second": bytes_bootstrap,
            "paired_characters_per_second": characters_bootstrap,
            "gates": gates,
            "quality_qualified_dominance_pass": not qualified or all(gates.values()),
        }

    cpu_gpu_identity = _identity(cpu_product, gpu_rows["ABI"])
    cpu_gates_recomputed = all(cpu_gates.values())
    gpu_quality_dominance = all(
        comparisons[system]["quality_qualified_dominance_pass"]
        for system in SYSTEMS[1:]
    )
    gates = {
        "all_gpu_run_gates_pass": all(
            result["status"].startswith("PASS") for result in gpu_results.values()
        ),
        "cpu_runtime_gates_pass": cpu["status"].startswith("PASS")
        and cpu_gates_recomputed,
        "cpu_gpu_candidate_output_identity": cpu_gpu_identity == 120,
        "quality_qualified_gpu_dominance": bool(quality_qualified)
        and gpu_quality_dominance,
        "same_candidate_seed": int(gpu_candidate["seed"])
        == int(protocol["candidate_seed"]),
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_B50_CPU_GPU_SAME_ARTIFACT_RUNTIME_COMPOSED"
        if all(gates.values())
        else "FAIL_B50_CPU_GPU_RUNTIME_GATE_CLOSED",
        "protocol_sha256": protocol_sha,
        "candidate_seed": int(protocol["candidate_seed"]),
        "quality_qualified_gpu_baselines": quality_qualified,
        "gpu_comparisons": comparisons,
        "cpu": {
            "status": cpu["status"],
            "comparisons": cpu["comparisons"],
            "gates": cpu["gates"],
        },
        "cpu_gpu_candidate_output_identities": cpu_gpu_identity,
        "gates": gates,
        "training_performed": False,
        "model_inference_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "claim_boundary": "Independent same-artifact CPU/GPU runtime composition on development prompts. The mixed B40 boundary, final test, Phase 4, and unconditional ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, (root / args.protocol).resolve(), (root / args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
