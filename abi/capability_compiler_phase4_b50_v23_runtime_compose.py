"""Independently compose exact-B50 v23 CPU/GPU runtime evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_gpu_runtime import SYSTEMS, _runtime_metrics
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_b50_runtime_compose import (
    _gpu_result,
    _identity,
    _jsonl,
    _metrics_equal,
    _paired_or_zero,
    _ratio,
    _recompute_cpu_evidence,
)
from .capability_compiler_phase4_b50_cpu_runtime import (
    RESULT_FORMAT as CPU_RESULT_FORMAT,
    _paired_prompt_throughput,
)
from .capability_compiler_phase4_b50_v23_runtime import (
    RESULT_FORMAT as V23_RESULT_FORMAT,
    load_cpu_protocol,
    load_gpu_protocol,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-v23-runtime-compose/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v23-runtime-compose-result/1"


def _wrapper_digest_valid(result: Mapping[str, Any]) -> bool:
    expected = str(result.get("evidence_sha256", ""))
    payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == expected


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_INDEPENDENT_EXACT_B50_V23_RUNTIME_COMPOSITION"
        or protocol.get("training_authorized") is not False
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or set(protocol.get("baseline_gpu_results", {})) != set(SYSTEMS[1:])
    ):
        raise Phase3Error("exact B50 v23 runtime composition governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 v23 composition binding changed: {relative}")
    return protocol, sha256_file(path)


def _load_wrapper(
    root: Path,
    path: Path,
    *,
    mode: str,
    protocol_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wrapper = _json(path)
    engine_path = root / str(wrapper["engine_result"])
    if (
        wrapper.get("format") != V23_RESULT_FORMAT
        or wrapper.get("mode") != mode
        or wrapper.get("runtime_interface") != "lc-direct-neural-core/23"
        or wrapper.get("protocol_sha256") != protocol_sha
        or not _wrapper_digest_valid(wrapper)
        or sha256_file(engine_path) != wrapper.get("engine_result_sha256")
        or sha256_file(root / str(wrapper["observations"]))
        != wrapper.get("observations_sha256")
        or wrapper.get("training_performed") is not False
        or wrapper.get("teacher_query_performed") is not False
        or wrapper.get("final_test_accessed") is not False
    ):
        raise Phase3Error(f"exact B50 v23 {mode} wrapper changed")
    engine = _json(engine_path)
    if engine != wrapper.get("engine") or not result_evidence_digest_valid(engine):
        raise Phase3Error(f"exact B50 v23 {mode} engine changed")
    return wrapper, engine


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    cpu_runtime, cpu_protocol_sha = load_cpu_protocol(
        root, root / str(protocol["candidate_cpu_protocol"])
    )
    gpu_runtime, gpu_protocol_sha = load_gpu_protocol(
        root, root / str(protocol["candidate_gpu_protocol"])
    )
    base_gpu_protocol_path = root / str(protocol["baseline_gpu_protocol"])
    base_gpu_protocol = _json(base_gpu_protocol_path)
    base_gpu_protocol_sha = sha256_file(base_gpu_protocol_path)

    quality = _json(root / str(protocol["matched_quality_result"]))
    if (
        quality.get("status")
        != "PASS_READ_ONLY_MATCHED_B50_QUALITY_AND_COST_RECOMPUTED"
        or not result_evidence_digest_valid(quality)
    ):
        raise Phase3Error("exact B50 matched-quality prerequisite changed")

    cpu_wrapper, cpu = _load_wrapper(
        root,
        root / str(protocol["candidate_cpu_result"]),
        mode="cpu",
        protocol_sha=cpu_protocol_sha,
    )
    if cpu.get("format") != CPU_RESULT_FORMAT:
        raise Phase3Error("exact B50 v23 CPU engine format changed")
    cpu_rows = _jsonl(root / str(cpu_wrapper["observations"]))
    cpu_product = [
        row for row in cpu_rows
        if row.get("system") == "layercake_v22_b50" and row.get("mode") == "product"
    ]
    cpu_ordinary = [
        row for row in cpu_rows
        if row.get("system") == "layercake_v22_b50" and row.get("mode") == "ordinary"
    ]
    cpu_qwen = [row for row in cpu_rows if row.get("system") == "qwen"]
    if not all(len(rows) == 120 for rows in (cpu_product, cpu_ordinary, cpu_qwen)):
        raise Phase3Error("exact B50 v23 CPU observation depth changed")
    if (
        not _metrics_equal(cpu["candidate"], _runtime_metrics(cpu_product))
        or not _metrics_equal(
            cpu["candidate"]["ordinary"], _runtime_metrics(cpu_ordinary)
        )
        or not _metrics_equal(
            cpu["optimized_transformer"], _runtime_metrics(cpu_qwen)
        )
    ):
        raise Phase3Error("exact B50 v23 CPU aggregates changed")
    cpu_gates, cpu_comparisons = _recompute_cpu_evidence(
        root, cpu_runtime, cpu, cpu_product, cpu_ordinary, cpu_qwen
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
        raise Phase3Error("exact B50 v23 CPU gates changed")

    gpu_wrapper, _ = _load_wrapper(
        root,
        root / str(protocol["candidate_gpu_result"]),
        mode="gpu",
        protocol_sha=gpu_protocol_sha,
    )
    gpu_results: dict[str, dict[str, Any]] = {}
    gpu_rows: dict[str, list[dict[str, Any]]] = {}
    gpu_results["ABI"], gpu_rows["ABI"] = _gpu_result(
        root,
        root / str(gpu_wrapper["engine_result"]),
        expected_system="ABI",
        expected_protocol_sha256=gpu_protocol_sha,
        runtime_protocol=gpu_runtime,
    )
    for system in SYSTEMS[1:]:
        gpu_results[system], gpu_rows[system] = _gpu_result(
            root,
            root / str(protocol["baseline_gpu_results"][system]),
            expected_system=system,
            expected_protocol_sha256=base_gpu_protocol_sha,
            runtime_protocol=base_gpu_protocol,
        )

    candidate = gpu_results["ABI"]
    comparisons: dict[str, Any] = {}
    quality_qualified: list[str] = []
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
        prompt_candidate_chars, prompt_baseline_chars = _paired_prompt_throughput(
            [{**row, "bytes_per_second": row["characters_per_second"]} for row in gpu_rows["ABI"]],
            [{**row, "bytes_per_second": row["characters_per_second"]} for row in gpu_rows[system]],
        )
        chars_bootstrap = _paired_or_zero(
            prompt_candidate_chars,
            prompt_baseline_chars,
            replicates=int(protocol["statistics"]["bootstrap_replicates"]),
            seed=int(protocol["statistics"]["characters_seed_base"]) + index,
        )
        for bootstrap, unit in ((bytes_bootstrap, "bytes"), (chars_bootstrap, "characters")):
            bootstrap["method"] = f"paired_prompt_median_{unit}_throughput_ratio_percentile_bootstrap"
            bootstrap["prompt_pairs"] = len(prompt_candidate_bytes)
        decision = quality["system_decisions"][system]
        qualified = bool(decision["baseline_all_seed_quality_pass"])
        if qualified:
            quality_qualified.append(system)
        ratios = {
            "median_bytes_per_second": _ratio(candidate["metrics"]["median_bytes_per_second"], baseline["metrics"]["median_bytes_per_second"]),
            "median_characters_per_second": _ratio(candidate["metrics"]["median_characters_per_second"], baseline["metrics"]["median_characters_per_second"]),
            "median_ttft": _ratio(candidate["metrics"]["median_time_to_first_output_seconds"], baseline["metrics"]["median_time_to_first_output_seconds"]),
            "active_tensor_bytes": _ratio(candidate["active_tensor_bytes"], baseline["active_tensor_bytes"]),
            "peak_process_rss": _ratio(candidate["peak_process_rss_delta_bytes"], baseline["peak_process_rss_delta_bytes"]),
            "peak_cuda": _ratio(candidate["peak_cuda_allocated_bytes"], baseline["peak_cuda_allocated_bytes"]),
            "cold_ttft": _ratio(candidate["cold"]["time_to_first_output_from_cold_start_seconds"], baseline["cold"]["time_to_first_output_from_cold_start_seconds"]),
        }
        gates = {
            "both_runtime_results_pass": candidate["status"].startswith("PASS") and baseline["status"].startswith("PASS"),
            "candidate_quality_noninferior": bool(decision["candidate_noninferior_all_seed"]),
            "median_gpu_throughput_not_slower": ratios["median_bytes_per_second"] is not None and ratios["median_bytes_per_second"] >= float(protocol["gates"]["quality_qualified_gpu_throughput_ratio_minimum"]),
            "paired_gpu_throughput_not_slower": bytes_bootstrap["lower_95"] is not None and bytes_bootstrap["lower_95"] >= float(protocol["gates"]["quality_qualified_gpu_paired_lower_minimum"]),
            "gpu_ttft_not_slower": ratios["median_ttft"] is not None and ratios["median_ttft"] <= float(protocol["gates"]["quality_qualified_gpu_ttft_ratio_maximum"]),
            "gpu_active_tensor_bytes_lower": ratios["active_tensor_bytes"] is not None and ratios["active_tensor_bytes"] < 1.0,
            "gpu_peak_rss_lower": ratios["peak_process_rss"] is not None and ratios["peak_process_rss"] < 1.0,
            "gpu_peak_cuda_lower": ratios["peak_cuda"] is not None and ratios["peak_cuda"] < 1.0,
        }
        comparisons[system] = {
            "baseline_quality_qualified": qualified,
            "ratios": ratios,
            "paired_bytes_per_second": bytes_bootstrap,
            "paired_characters_per_second": chars_bootstrap,
            "gates": gates,
            "quality_qualified_dominance_pass": qualified and all(gates.values()),
        }

    cpu_gpu_identity = _identity(cpu_product, gpu_rows["ABI"])
    gates = {
        "all_gpu_run_gates_pass": all(row["status"].startswith("PASS") for row in gpu_results.values()),
        "cpu_runtime_gates_pass": cpu_wrapper["status"].startswith("PASS") and all(cpu_gates.values()),
        "cpu_gpu_candidate_output_identity": cpu_gpu_identity == 120,
        "quality_qualified_gpu_comparator_exists": bool(quality_qualified),
        "quality_qualified_gpu_dominance": bool(quality_qualified) and all(comparisons[system]["quality_qualified_dominance_pass"] for system in quality_qualified),
        "same_candidate_seed": int(candidate["seed"]) == int(protocol["candidate_seed"]),
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_EXACT_B50_V23_CPU_GPU_RUNTIME_COMPOSED" if all(gates.values()) else "FAIL_EXACT_B50_V23_RUNTIME_GATE_CLOSED",
        "protocol_sha256": protocol_sha,
        "candidate_seed": int(protocol["candidate_seed"]),
        "candidate_package": dict(protocol["candidate_package"]),
        "quality_qualified_gpu_baselines": quality_qualified,
        "gpu_comparisons": comparisons,
        "cpu": {"status": cpu_wrapper["status"], "comparisons": cpu_comparisons, "gates": cpu_gates},
        "cpu_gpu_candidate_output_identities": cpu_gpu_identity,
        "gates": gates,
        "training_performed": False,
        "model_inference_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "abi_superiority_certified": False,
        "claim_boundary": "Independent exact-V23 same-artifact CPU/GPU runtime composition on development prompts. CPU gates, a quality-qualified matched comparator, final test, Phase 4, and unconditional ABI superiority remain open unless explicitly passed here.",
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
    result = verify(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
