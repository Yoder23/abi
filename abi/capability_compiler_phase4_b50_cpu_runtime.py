"""Measure verified B50 v22 CPU runtime against pinned optimized Qwen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Iterable, Mapping

import psutil
import torch

from . import capability_compiler_phase3_cpu_runtime as runtime
from .capability_compiler_phase2_common import (
    canonical_json_bytes,
    evaluate_functional,
    sha256_file,
)
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase3_cpu_runtime_v2 import _ps_model, force_cpu_body
from .capability_compiler_phase3_qwen_rss_audit import runner_working_set
from .capability_compiler_phase4_b50_gpu_runtime import (
    _build_candidate,
    _candidate_request,
    _identity,
    _observation,
    _reference,
    _runtime_metrics,
    _tensor_bytes,
    runtime_schedule,
)
from .capability_compiler_phase4_v19_cpu_runtime import (
    _tags_digest,
    paired_quality_bootstrap,
)
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-b50-cpu-runtime/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-cpu-runtime-result/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    cfg = protocol.get("runtime", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_SAME_ARTIFACT_B50_V22_CPU_RUNTIME"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_generation_authorized") is not False
        or protocol.get("candidate_construction_authorized") is not False
        or protocol.get("deterministic_package_rebuild_authorized") is not True
        or protocol.get("final_test_access") != "PROHIBITED"
        or int(cfg.get("distinct_prompts", 0)) != 100
        or int(cfg.get("repeated_observations", 0)) < 20
        or int(cfg.get("torch_threads", 0)) != 1
        or int(cfg.get("ollama_num_gpu", -1)) != 0
    ):
        raise Phase3Error("matched B50 CPU runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"matched B50 CPU runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def _qwen_probe(probe: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "probe_id": str(probe["probe_id"]),
        "prompt": str(probe["prompt"]),
        "max_new_tokens": int(probe["max_new_tokens"]),
    }


def _ordinary_request(host: Any, probe: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    state = host.prefill(str(probe["prompt"]))
    first = None
    for _ in range(int(probe["max_new_tokens"])):
        token = host.decode_step(state)
        if token is None:
            break
        if first is None:
            first = time.perf_counter() - started
    output = host.realize(state).decode("utf-8", errors="strict")
    total = time.perf_counter() - started
    token_ids = [int(value) for value in host.model_tokenizer.encode(output)]
    return _observation(
        probe=probe,
        output=output,
        output_token_ids=token_ids,
        first_seconds=float(first if first is not None else total),
        total_seconds=total,
        execution={
            "guard_terminated": bool(state["terminated_by_guard"]),
            "active_residual_routes": 0 if int(state["weak_route"]) < 0 else 1,
            "persistent_state_created": state["past_key_values"] is not None,
            "ordinary_direct_path": True,
        },
    )


def _paired_prompt_throughput(
    candidate: list[Mapping[str, Any]], baseline: list[Mapping[str, Any]]
) -> tuple[list[float], list[float]]:
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    if len(candidate) != len(baseline):
        raise Phase3Error("paired runtime observation depth changed")
    for left, right in zip(candidate, baseline):
        probe_id = str(left["probe_id"])
        if probe_id != str(right["probe_id"]):
            raise Phase3Error("paired runtime prompt order changed")
        left_values, right_values = grouped.setdefault(probe_id, ([], []))
        left_values.append(float(left["bytes_per_second"]))
        right_values.append(float(right["bytes_per_second"]))
    ordered = sorted(grouped)
    return (
        [statistics.median(grouped[key][0]) for key in ordered],
        [statistics.median(grouped[key][1]) for key in ordered],
    )


def _paired_ratio_or_zero(
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
        **runtime.paired_ratio_bootstrap(candidate, baseline, replicates, seed),
        "zero_candidate_observations": 0,
        "zero_baseline_observations": 0,
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    distinct, scheduled = runtime_schedule(root, protocol)
    baseline = protocol["transformer_baseline"]
    digest = _tags_digest(str(baseline["base_url"]), str(baseline["model"]))
    gates = {
        "depth": len(distinct) == 100 and len(scheduled) == 120,
        "schedule_bound": hashlib.sha256(
            "\n".join(str(row["probe_id"]) for row in scheduled).encode("utf-8")
        ).hexdigest()
        == protocol["runtime"]["schedule_sha256"],
        "qwen_digest_bound": digest == baseline["digest"],
        "cuda_initially_unused": int(torch.cuda.memory_allocated()) == 0,
        "training_prohibited": True,
        "teacher_query_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "status": "PASS_B50_V22_CPU_RUNTIME_PREFLIGHT"
        if all(gates.values())
        else "FAIL_B50_V22_CPU_RUNTIME_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
        "model_inference_performed": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable matched B50 CPU runtime output exists: {output}")
    if not preflight(root, protocol_path)["status"].startswith("PASS"):
        raise Phase3Error("matched B50 CPU runtime preflight failed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    distinct, scheduled = runtime_schedule(root, protocol)
    candidate_spec = protocol["systems"]["ABI"]
    reference = _reference(
        root / str(candidate_spec["quality_reference_outputs"]),
        str(candidate_spec["quality_reference_sha256"]),
    )
    process = psutil.Process()
    process_initial_rss = process.memory_info().rss
    candidate_cuda_before = int(torch.cuda.memory_allocated())
    temporary = tempfile.TemporaryDirectory(prefix="abi-b50-v22-cpu-")
    temporary_path = Path(temporary.name)
    package_started = time.perf_counter()
    with runtime.PeakMonitor(lambda: process.memory_info().rss) as package_monitor:
        package_path, package, public_pem, signer, api, _ = _build_candidate(
            root, protocol, temporary_path
        )
    package_build_seconds = time.perf_counter() - package_started
    package_peak_rss_delta = max(0, package_monitor.peak - process_initial_rss)
    gc.collect()
    runtime_idle_rss = process.memory_info().rss
    with runtime.PeakMonitor(lambda: process.memory_info().rss) as candidate_monitor:
        cold_started = time.perf_counter()
        host = api["Host"](
            temporary_path / "registry",
            trust_store={signer: public_pem},
            device="cpu",
        )
        activation = host.activate(package_path)
        model_load_seconds = time.perf_counter() - cold_started
        candidate_cold = _candidate_request(host, distinct[0])
        candidate_cold["single_cold_request"] = True
        candidate_cold["cold_definition"] = (
            "model-residency cold; exactly one generation request after host/model "
            "load; operating-system filesystem cache was not purged"
        )
        candidate_cold["model_load_seconds"] = model_load_seconds
        candidate_cold["time_to_first_output_from_cold_start_seconds"] = (
            model_load_seconds + candidate_cold["time_to_first_output_seconds"]
        )
        candidate_cold["total_from_cold_start_seconds"] = (
            time.perf_counter() - cold_started
        )
        for probe in distinct[: int(protocol["runtime"]["warmup_observations"])]:
            _candidate_request(host, probe)
        candidate_rows = [_candidate_request(host, probe) for probe in scheduled]
        for probe in distinct[: int(protocol["runtime"]["warmup_observations"])]:
            _ordinary_request(host, probe)
        ordinary_rows = [_ordinary_request(host, probe) for probe in scheduled]
    candidate_peak_rss_delta = max(
        0, candidate_monitor.peak - runtime_idle_rss
    )
    candidate_active_tensor_bytes = sum(
        _tensor_bytes(module) for module in (host.model, host.router, host.residual)
    )
    candidate_cuda_after = int(torch.cuda.memory_allocated())
    candidate_cuda_peak = int(torch.cuda.max_memory_allocated())
    candidate_identity = _identity(candidate_rows, reference)

    base_url = str(protocol["transformer_baseline"]["base_url"])
    model = str(protocol["transformer_baseline"]["model"])
    runtime._ollama_unload(base_url, model)
    if runner_working_set() != 0:
        raise Phase3Error("llama-server remained resident after matched B50 unload")
    original_post = runtime._post_json
    device_records: list[dict[str, Any]] = []

    def patched_post(url: str, body: Mapping[str, Any], *, stream: bool = False):
        return original_post(url, force_cpu_body(url, body), stream=stream)

    runtime._post_json = patched_post
    try:
        with runtime.PeakMonitor(runner_working_set) as qwen_monitor:
            qwen_cold = runtime._ollama_request(
                base_url,
                model,
                _qwen_probe(distinct[0]),
                str(protocol["runtime"]["keep_alive"]),
            )
            qwen_cold["single_cold_request"] = True
            qwen_cold["cold_definition"] = (
                "Ollama model-residency cold after an unload control; exactly one "
                "streaming generation request; operating-system filesystem cache "
                "was not purged"
            )
            qwen_cold["token_accounting"] = "authoritative_runtime_eval_count"
            device_records.append(_ps_model(base_url, model))
            for probe in distinct[: int(protocol["runtime"]["warmup_observations"])]:
                runtime._ollama_request(
                    base_url,
                    model,
                    _qwen_probe(probe),
                    str(protocol["runtime"]["keep_alive"]),
                )
            qwen_rows = []
            for probe in scheduled:
                row = runtime._ollama_request(
                    base_url,
                    model,
                    _qwen_probe(probe),
                    str(protocol["runtime"]["keep_alive"]),
                )
                row["token_accounting"] = "authoritative_runtime_eval_count"
                row["functional_pass"] = evaluate_functional(
                    str(row["output"]), probe["evaluator"]
                )
                row["functional_pass_v2"] = evaluate_functional_v2(
                    str(row["output"]),
                    probe["evaluator"],
                    str(probe["canonical_capability"]),
                )
                row["repetition_collapse_v2"] = repetition_collapse_v2(
                    str(row["output"])
                )
                row["capability"] = str(probe["canonical_capability"])
                qwen_rows.append(row)
                device_records.append(_ps_model(base_url, model))
        qwen_peak_rss = qwen_monitor.peak
    finally:
        runtime._post_json = original_post
        runtime._ollama_unload(base_url, model)

    candidate_metrics = _runtime_metrics(candidate_rows)
    ordinary_metrics = _runtime_metrics(ordinary_rows)
    qwen_metrics = _runtime_metrics(qwen_rows)
    prompt_candidate_throughput, prompt_qwen_throughput = _paired_prompt_throughput(
        candidate_rows, qwen_rows
    )
    paired_speed = _paired_ratio_or_zero(
        prompt_candidate_throughput,
        prompt_qwen_throughput,
        replicates=int(protocol["statistics"]["bootstrap_replicates"]),
        seed=int(protocol["statistics"]["throughput_bootstrap_seed"]),
    )
    paired_speed["method"] = (
        "paired_prompt_median_throughput_ratio_percentile_bootstrap"
    )
    paired_speed["prompt_pairs"] = len(prompt_candidate_throughput)
    candidate_quality_v1 = [
        evaluate_functional(str(row["output"]), probe["evaluator"])
        for row, probe in zip(candidate_rows[: len(distinct)], distinct)
    ]
    candidate_quality_v2 = [
        evaluate_functional_v2(
            str(row["output"]),
            probe["evaluator"],
            str(probe["canonical_capability"]),
        )
        for row, probe in zip(candidate_rows[: len(distinct)], distinct)
    ]
    qwen_quality_v1 = [
        bool(row["functional_pass"]) for row in qwen_rows[: len(distinct)]
    ]
    qwen_quality_v2 = [
        bool(row["functional_pass_v2"]) for row in qwen_rows[: len(distinct)]
    ]
    paired_quality_v1 = paired_quality_bootstrap(
        candidate_quality_v1,
        qwen_quality_v1,
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["quality_bootstrap_seed"]),
    )
    paired_quality_v2 = paired_quality_bootstrap(
        candidate_quality_v2,
        qwen_quality_v2,
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["quality_bootstrap_seed"]) + 1,
    )
    throughput_ratio = (
        candidate_metrics["median_bytes_per_second"]
        / qwen_metrics["median_bytes_per_second"]
    )
    ttft_ratio = (
        candidate_metrics["median_time_to_first_output_seconds"]
        / qwen_metrics["median_time_to_first_output_seconds"]
    )
    cold_total_ratio = (
        candidate_cold["total_from_cold_start_seconds"]
        / qwen_cold["total_seconds"]
    )
    retention = (
        ordinary_metrics["median_bytes_per_second"]
        / float(protocol["locked_phase2_runtime"]["median_bytes_per_second"])
    )
    gates_cfg = protocol["gates"]
    pointer_rows = [
        row for row in candidate_rows if row["capability"] == "coherence"
    ]
    format_rows = [
        row for row in candidate_rows if row["capability"] == "format_control"
    ]
    ordinary_product_rows = [
        row
        for row in candidate_rows
        if row["capability"] not in {"coherence", "format_control"}
    ]
    gates = {
        "same_signed_package": activation["archive_hash"]
        == candidate_spec["archive_sha256"],
        "payload_preserved": activation["payload_hash"]
        == candidate_spec["tensor_payload_hash"],
        "runtime_outputs_exact_to_quality_candidate": candidate_identity
        == len(candidate_rows),
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
        "cold_ttft_no_worse": candidate_cold[
            "time_to_first_output_from_cold_start_seconds"
        ]
        <= qwen_cold["time_to_first_output_seconds"],
        "lower_active_tensor_bytes": candidate_active_tensor_bytes
        < int(protocol["transformer_baseline"]["model_file_bytes"]),
        "lower_peak_active_rss": candidate_peak_rss_delta < qwen_peak_rss,
        "candidate_fully_cpu": candidate_cuda_before
        == candidate_cuda_after
        == candidate_cuda_peak
        == 0,
        "qwen_fully_cpu": len(device_records) == 121
        and all(int(row.get("size_vram", -1)) == 0 for row in device_records),
        "genuine_candidate_cold_single_request": candidate_cold[
            "single_cold_request"
        ]
        and candidate_cold["model_load_seconds"] > 0,
        "genuine_qwen_cold_single_request": qwen_cold["single_cold_request"]
        and qwen_cold["load_seconds_reported"] > 0,
        "authoritative_token_accounting": all(
            row["token_accounting"] == "completed_response_retokenization"
            and row["authoritative_output_tokens"]
            == len(row["retokenized_output_token_ids"])
            for row in candidate_rows + ordinary_rows
        )
        and all(
            row["token_accounting"] == "authoritative_runtime_eval_count"
            and row["authoritative_output_tokens"] >= 0
            for row in qwen_rows
        ),
        "depth": len(candidate_rows) == len(qwen_rows) == 120
        and len({row["probe_id"] for row in candidate_rows}) == 100
        and paired_speed["prompt_pairs"] == 100,
        "candidate_zero_repetition_collapse_v2": not any(
            repetition_collapse_v2(str(row["output"])) for row in candidate_rows
        ),
        "p95_supported": candidate_metrics["p95_supported"]
        and qwen_metrics["p95_supported"],
        "p99_not_promoted": not candidate_metrics["p99_supported"]
        and not qwen_metrics["p99_supported"],
        "receiver_learning_zero": activation["receiver_training_steps"]
        == activation["receiver_calibration_runs"]
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
            and row["execution"]["format"].get(
                "candidate_scoring_forward_passes"
            )
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
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    output.mkdir(parents=True)
    observations = output / "observations.jsonl"
    _write_immutable(
        observations,
        b"".join(
            canonical_json_bytes({"system": system, "mode": mode, **row})
            for system, mode, rows in (
                ("layercake_v22_b50", "product", candidate_rows),
                ("layercake_v22_b50", "ordinary", ordinary_rows),
                ("qwen", "product", qwen_rows),
            )
            for row in rows
        ),
    )
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_SAME_ARTIFACT_B50_V22_CPU_RUNTIME"
        if all(gates.values())
        else "FAIL_SAME_ARTIFACT_B50_V22_CPU_RUNTIME",
        "protocol_sha256": protocol_sha,
        "package": {
            **package,
            "package_build_seconds_one_time_not_in_request": package_build_seconds,
            "package_build_peak_rss_delta_bytes_one_time": package_peak_rss_delta,
        },
        "candidate": {
            **candidate_metrics,
            "functional_passes_v1_distinct": sum(candidate_quality_v1),
            "functional_passes_v2_distinct": sum(candidate_quality_v2),
            "active_tensor_bytes": candidate_active_tensor_bytes,
            "peak_active_rss_delta_bytes": candidate_peak_rss_delta,
            "runtime_rss_baseline_bytes": runtime_idle_rss,
            "quality_output_identities": candidate_identity,
            "cold": candidate_cold,
            "ordinary": {
                **ordinary_metrics,
                "throughput_retention": retention,
            },
        },
        "optimized_transformer": {
            **qwen_metrics,
            "functional_passes_v1_distinct": sum(qwen_quality_v1),
            "functional_passes_v2_distinct": sum(qwen_quality_v2),
            "model": model,
            "digest": protocol["transformer_baseline"]["digest"],
            "model_file_bytes": protocol["transformer_baseline"]["model_file_bytes"],
            "peak_runner_rss_bytes": qwen_peak_rss,
            "cold": qwen_cold,
        },
        "comparisons": {
            "median_throughput_ratio": throughput_ratio,
            "paired_throughput": paired_speed,
            "paired_quality_v1": paired_quality_v1,
            "paired_quality_v2": paired_quality_v2,
            "median_ttft_ratio": ttft_ratio,
            "cold_total_latency_ratio": cold_total_ratio,
            "peak_rss_ratio": candidate_peak_rss_delta / qwen_peak_rss,
            "ordinary_phase2_throughput_retention": retention,
        },
        "device_control": {
            "candidate_cuda_allocated_before_bytes": candidate_cuda_before,
            "candidate_cuda_allocated_after_bytes": candidate_cuda_after,
            "candidate_cuda_peak_allocated_bytes": candidate_cuda_peak,
            "qwen_size_vram_zero_observations": sum(
                int(row.get("size_vram", -1)) == 0 for row in device_records
            ),
            "qwen_device_observations": len(device_records),
        },
        "prompt_depth": {"distinct": 100, "repeated": 20, "observations": 120},
        "hardware": {
            "machine": platform.node(),
            "logical_cpu_count": psutil.cpu_count(),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "memory_accounting": {
            "candidate_active_tensor_bytes": "parameters and buffers resident in the executing LayerCake modules",
            "candidate_runtime_peak_rss_delta": "peak process RSS during host load and inference minus post-packaging pre-load process RSS",
            "candidate_package_build_peak_rss_delta": "separately reported one-time deterministic packaging peak; excluded from inference RSS",
            "qwen_peak_runner_rss_bytes": "absolute working set of the freshly loaded Ollama runner process",
        },
        "gates": gates,
        "raw_observations_sha256": sha256_file(observations),
        "teacher_present": False,
        "training_performed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Same-artifact B50 v22 CPU runtime versus pinned optimized CPU Qwen plus ordinary-host retention. No final test, stable minimum, Phase 4, or unconditional ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    temporary.cleanup()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("run")
    command.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = (
        preflight(root, (root / args.protocol).resolve())
        if args.command == "preflight"
        else run(
            root,
            (root / args.protocol).resolve(),
            (root / args.output_dir).resolve(),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
