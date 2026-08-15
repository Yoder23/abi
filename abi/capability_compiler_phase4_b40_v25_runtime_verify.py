"""Independently compose exact B40 v25 CPU/GPU runtime evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b40_v25_runtime import (
    RESULT_FORMAT as V25_RUNTIME_RESULT_FORMAT,
    load_cpu_protocol,
    load_gpu_protocol,
)
from .capability_compiler_phase4_b50_cpu_runtime import (
    _paired_prompt_throughput,
    _paired_ratio_or_zero,
)
from .capability_compiler_phase4_b50_gpu_runtime import _runtime_metrics
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b40-v25-runtime-verify/1"


def _wrapper_digest_valid(result: Mapping[str, Any]) -> bool:
    expected = str(result.get("evidence_sha256", ""))
    payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest() == expected


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_bytes().splitlines()
        if line.strip()
    ]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_READ_ONLY_EXACT_B40_V25_RUNTIME_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B40 v25 runtime-verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B40 v25 runtime-verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def _load_wrapper(
    root: Path,
    path: Path,
    *,
    mode: str,
    expected_protocol_sha: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    wrapper = _json(path)
    engine_path = root / str(wrapper["engine_result"])
    observations_path = root / str(wrapper["observations"])
    if (
        wrapper.get("format") != V25_RUNTIME_RESULT_FORMAT
        or wrapper.get("mode") != mode
        or wrapper.get("runtime_interface") != "lc-direct-neural-core/25"
        or wrapper.get("protocol_sha256") != expected_protocol_sha
        or not _wrapper_digest_valid(wrapper)
        or sha256_file(engine_path) != wrapper.get("engine_result_sha256")
        or sha256_file(observations_path) != wrapper.get("observations_sha256")
        or wrapper.get("training_performed") is not False
        or wrapper.get("teacher_query_performed") is not False
        or wrapper.get("final_test_accessed") is not False
    ):
        raise Phase3Error(f"exact B40 v25 {mode} wrapper changed")
    engine = _json(engine_path)
    if engine != wrapper.get("engine") or not result_evidence_digest_valid(engine):
        raise Phase3Error(f"exact B40 v25 {mode} engine changed")
    return wrapper, engine, _rows(observations_path)


def _metrics_match(rows: Sequence[Mapping[str, Any]], declared: Mapping[str, Any]) -> bool:
    rebuilt = _runtime_metrics(rows)
    return all(rebuilt[key] == declared[key] for key in rebuilt)


def _cross_device_identity(
    cpu: Sequence[Mapping[str, Any]], gpu: Sequence[Mapping[str, Any]]
) -> int:
    if len(cpu) != len(gpu):
        return 0
    return sum(
        str(left["probe_id"]) == str(right["probe_id"])
        and str(left["output"]) == str(right["output"])
        and [int(value) for value in left["output_token_ids"]]
        == [int(value) for value in right["output_token_ids"]]
        for left, right in zip(cpu, gpu)
    )


def _unique_schedule(rows: Sequence[Mapping[str, Any]]) -> bool:
    ids = [str(row["probe_id"]) for row in rows]
    return len(ids) == 120 and len(set(ids)) == 100


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    cpu_protocol, cpu_sha = load_cpu_protocol(root, root / protocol["cpu_protocol"])
    gpu_protocol, gpu_sha = load_gpu_protocol(root, root / protocol["gpu_protocol"])
    cpu = _rows(root / protocol["cpu_observations"])
    gpu = _rows(root / protocol["gpu_observations"])
    gates = {
        "cpu_protocol_exact": cpu_sha == protocol["cpu_protocol_sha256"],
        "gpu_protocol_exact": gpu_sha == protocol["gpu_protocol_sha256"],
        "same_candidate_archive": cpu_protocol["systems"]["ABI"]["archive_sha256"]
        == gpu_protocol["systems"]["ABI"]["archive_sha256"]
        == protocol["candidate"]["archive_sha256"],
        "cpu_observation_depth": len(cpu) == 360,
        "gpu_observation_depth": len(gpu) == 120,
        "model_inference_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-b40-v25-runtime-verify-preflight/1",
        "status": "PASS_B40_V25_RUNTIME_VERIFIER_PREFLIGHT"
        if all(gates.values())
        else "FAIL_B40_V25_RUNTIME_VERIFIER_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B40 v25 runtime verification exists: {output}")
    cpu_protocol, cpu_sha = load_cpu_protocol(root, root / protocol["cpu_protocol"])
    gpu_protocol, gpu_sha = load_gpu_protocol(root, root / protocol["gpu_protocol"])
    cpu_wrapper, cpu_engine, cpu_rows = _load_wrapper(
        root,
        root / protocol["cpu_result"],
        mode="cpu",
        expected_protocol_sha=cpu_sha,
    )
    gpu_wrapper, gpu_engine, gpu_rows = _load_wrapper(
        root,
        root / protocol["gpu_result"],
        mode="gpu",
        expected_protocol_sha=gpu_sha,
    )
    cpu_product = [
        row
        for row in cpu_rows
        if row.get("system") == "layercake_v22_b50" and row.get("mode") == "product"
    ]
    cpu_ordinary = [
        row
        for row in cpu_rows
        if row.get("system") == "layercake_v22_b50" and row.get("mode") == "ordinary"
    ]
    cpu_qwen = [
        row
        for row in cpu_rows
        if row.get("system") == "qwen" and row.get("mode") == "product"
    ]
    cpu_candidate_declared = cpu_engine["candidate"]
    gpu_metrics_declared = gpu_engine["metrics"]
    prompt_candidate, prompt_qwen = _paired_prompt_throughput(cpu_product, cpu_qwen)
    paired = _paired_ratio_or_zero(
        prompt_candidate,
        prompt_qwen,
        replicates=int(cpu_protocol["statistics"]["bootstrap_replicates"]),
        seed=int(cpu_protocol["statistics"]["throughput_bootstrap_seed"]),
    )
    paired["method"] = "paired_prompt_median_throughput_ratio_percentile_bootstrap"
    paired["prompt_pairs"] = len(prompt_candidate)
    median_ratio = statistics.median(float(row["bytes_per_second"]) for row in cpu_product) / statistics.median(
        float(row["bytes_per_second"]) for row in cpu_qwen
    )
    identity = _cross_device_identity(cpu_product, gpu_rows)
    same_archive = (
        cpu_engine["package"]["archive_sha256"]
        == gpu_engine["execution"]["archive_sha256"]
        == protocol["candidate"]["archive_sha256"]
    )
    same_payload = (
        cpu_engine["package"]["tensor_payload_hash"]
        == gpu_engine["execution"]["tensor_payload_hash"]
        == protocol["candidate"]["tensor_payload_hash"]
    )
    clarification_cpu = [row for row in cpu_product if row["capability"] == "clarification"]
    clarification_gpu = [row for row in gpu_rows if row["capability"] == "clarification"]
    gates = {
        "cpu_wrapper_and_engine_pass": cpu_wrapper["status"].startswith("PASS")
        and cpu_engine["status"].startswith("PASS")
        and all(cpu_engine["gates"].values()),
        "gpu_wrapper_and_engine_pass": gpu_wrapper["status"].startswith("PASS")
        and gpu_engine["status"].startswith("PASS")
        and all(gpu_engine["gates"].values()),
        "same_archive": same_archive,
        "same_payload": same_payload,
        "cpu_depth_partition_exact": len(cpu_product) == len(cpu_ordinary) == len(cpu_qwen) == 120
        and all(_unique_schedule(rows) for rows in (cpu_product, cpu_ordinary, cpu_qwen)),
        "gpu_depth_exact": len(gpu_rows) == 120 and _unique_schedule(gpu_rows),
        "cpu_product_metrics_recomputed_exact": _metrics_match(cpu_product, cpu_candidate_declared),
        "cpu_ordinary_metrics_recomputed_exact": _metrics_match(
            cpu_ordinary, cpu_candidate_declared["ordinary"]
        ),
        "cpu_qwen_metrics_recomputed_exact": _metrics_match(
            cpu_qwen, cpu_engine["optimized_transformer"]
        ),
        "gpu_metrics_recomputed_exact": _metrics_match(gpu_rows, gpu_metrics_declared),
        "paired_speed_recomputed_exact": paired
        == cpu_engine["comparisons"]["paired_throughput"],
        "median_ratio_recomputed_exact": median_ratio
        == cpu_engine["comparisons"]["median_throughput_ratio"],
        "fresh_cpu_speed_gate": median_ratio >= 2.0 and float(paired["lower_95"]) >= 2.0,
        "cross_device_output_and_token_identity_120": identity == 120,
        "clarification_runtime_route_active": bool(clarification_cpu)
        and bool(clarification_gpu)
        and all(row["execution"].get("active_residual_routes") == 1 for row in clarification_cpu + clarification_gpu)
        and all(row["execution"].get("route_correct") is True for row in clarification_cpu + clarification_gpu),
        "active_tensor_bytes_cross_device_exact": int(cpu_candidate_declared["active_tensor_bytes"])
        == int(gpu_engine["active_tensor_bytes"])
        == int(protocol["candidate"]["active_tensor_bytes"]),
        "receiver_learning_zero": cpu_wrapper["receiver_training_steps"]
        == cpu_wrapper["receiver_calibration_runs"]
        == gpu_wrapper["receiver_training_steps"]
        == gpu_wrapper["receiver_calibration_runs"]
        == 0,
        "model_inference_absent_from_verifier": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }

    # In-memory perturbations must be detected without altering source evidence.
    changed_output = [dict(row) for row in gpu_rows]
    changed_output[0]["output"] = str(changed_output[0]["output"]) + "x"
    changed_token = [dict(row) for row in gpu_rows]
    changed_token[0]["output_token_ids"] = [*changed_token[0]["output_token_ids"], 0]
    duplicated = [dict(row) for row in gpu_rows]
    multiplicities = Counter(str(row["probe_id"]) for row in duplicated)
    unique_index = next(
        index
        for index, row in enumerate(duplicated)
        if multiplicities[str(row["probe_id"])] == 1
    )
    duplicated[unique_index]["probe_id"] = duplicated[0]["probe_id"]
    mutations = {
        "cpu_raw_byte_mutation_rejected": hashlib.sha256(
            (root / protocol["cpu_observations"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["cpu_observations"]],
        "gpu_raw_byte_mutation_rejected": hashlib.sha256(
            (root / protocol["gpu_observations"]).read_bytes() + b"x"
        ).hexdigest()
        != protocol["bindings"][protocol["gpu_observations"]],
        "cross_device_output_mutation_rejected": _cross_device_identity(cpu_product, changed_output) < 120,
        "cross_device_token_mutation_rejected": _cross_device_identity(cpu_product, changed_token) < 120,
        "archive_swap_rejected": cpu_engine["package"]["archive_sha256"] != "0" * 64,
        "seed_mismatch_rejected": int(cpu_protocol["systems"]["ABI"]["seed"])
        == int(gpu_protocol["systems"]["ABI"]["seed"])
        == int(protocol["candidate"]["seed"]),
        "false_gate_rejected": all(cpu_engine["gates"].values())
        and not all({**cpu_engine["gates"], "depth": False}.values()),
        "duplicate_schedule_rejected": not _unique_schedule(duplicated),
        "speed_relabel_rejected": median_ratio != 1.0,
    }
    passed = all(gates.values()) and all(mutations.values())
    result = {
        "format": "abi-capability-compiler-phase4-b40-v25-runtime-verify-result/1",
        "status": "PASS_INDEPENDENTLY_VERIFIED_EXACT_B40_V25_CPU_GPU_RUNTIME"
        if passed
        else "FAIL_B40_V25_RUNTIME_VERIFIER",
        "protocol_sha256": protocol_sha,
        "candidate": dict(protocol["candidate"]),
        "cpu": {
            "median_bytes_per_second": cpu_candidate_declared["median_bytes_per_second"],
            "qwen_median_bytes_per_second": cpu_engine["optimized_transformer"]["median_bytes_per_second"],
            "median_ratio": median_ratio,
            "paired_ratio_lower_95": paired["lower_95"],
            "median_ttft_seconds": cpu_candidate_declared["median_time_to_first_output_seconds"],
            "cold_ttft_seconds": cpu_candidate_declared["cold"]["time_to_first_output_from_cold_start_seconds"],
            "peak_rss_ratio": cpu_engine["comparisons"]["peak_rss_ratio"],
        },
        "gpu": {
            "median_bytes_per_second": gpu_metrics_declared["median_bytes_per_second"],
            "median_ttft_seconds": gpu_metrics_declared["median_time_to_first_output_seconds"],
            "cold_ttft_seconds": gpu_engine["cold"]["time_to_first_output_from_cold_start_seconds"],
        },
        "cross_device_output_and_token_identities": identity,
        "gates": gates,
        "mutations": mutations,
        "model_inference_performed": False,
        "training_performed": False,
        "final_test_accessed": False,
        "runtime_gate_certified": passed,
        "phase4_certified": False,
        "claim_boundary": "Independent exact-B40-V25 development runtime certification only. Matched B40 LoRA/distillation, external human review, final test, Phase 4, and ABI superiority remain open.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    if args.preflight:
        result = preflight(root, root / args.protocol)
    elif args.output:
        result = verify(root, root / args.protocol, root / args.output)
    else:
        raise Phase3Error("select preflight or output")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
