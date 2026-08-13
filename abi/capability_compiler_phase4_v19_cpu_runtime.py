"""Repeated same-artifact CPU runtime for the v19 prompt-span path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
from urllib.request import urlopen

import psutil
import torch

from . import capability_compiler_phase3_cpu_runtime as runtime
from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_cpu_runtime_v2 import _ps_model, force_cpu_body
from .capability_compiler_phase3_qwen_rss_audit import runner_working_set


FORMAT = "abi-capability-compiler-phase4-v19-cpu-runtime/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(value, dict) for value in values):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return values


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    cfg = protocol.get("runtime", {})
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_V19_SAME_ARTIFACT_CPU_RUNTIME"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or int(cfg.get("distinct_prompts", 0)) < 100
        or int(cfg.get("repeated_observations", 0)) < 20
        or int(cfg.get("torch_threads", 0)) != 1
        or int(cfg.get("ollama_num_gpu", -1)) != 0
    ):
        raise Phase3Error("v19 CPU runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v19 CPU runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def select_prospective(rows: list[dict[str, Any]], distinct: int, repeats: int):
    ordered = sorted(rows, key=lambda row: str(row["ir_record_id"]))
    selected = ordered[:distinct]
    if len(selected) != distinct or len({row["ir_record_id"] for row in selected}) != distinct:
        raise Phase3Error("prospective runtime prompt selection changed")
    return selected, [*selected, *[selected[index % distinct] for index in range(repeats)]]


def paired_quality_bootstrap(candidate: list[bool], baseline: list[bool], replicates: int, seed: int):
    if len(candidate) != len(baseline) or not candidate:
        raise Phase3Error("invalid paired quality inputs")
    differences = [float(left) - float(right) for left, right in zip(candidate, baseline)]
    generator = random.Random(seed)
    draws = [statistics.mean(differences[generator.randrange(len(differences))] for _ in differences) for _ in range(replicates)]
    return {
        "method": "paired_prompt_level_quality_difference_percentile_bootstrap",
        "observations": len(differences),
        "replicates": replicates,
        "seed": seed,
        "point": statistics.mean(differences),
        "lower_95": runtime.percentile(draws, 0.025),
        "upper_95": runtime.percentile(draws, 0.975),
    }


def _host_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake_extensions.route_isolated_prompt_span_core_v19 import PromptSpanRouteIsolatedShallowSparseCoreHost
    return PromptSpanRouteIsolatedShallowSparseCoreHost


def _tensor_bytes(module: torch.nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in module.parameters()) + sum(value.numel() * value.element_size() for value in module.buffers())


def _pointer_request(host: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output = host.generate(str(row["normalized_generation_prompt"]), maximum_tokens=int(row["generation_max_new_tokens"])).decode("utf-8", errors="strict")
    total = time.perf_counter() - started
    raw = output.encode("utf-8")
    token_ids = host.model_tokenizer.encode(output)
    pointer = dict(host.last_pointer_execution or {})
    pointer.pop("wall_seconds", None)
    return {
        "probe_id": row["ir_record_id"],
        "output": output,
        "output_utf8_bytes": len(raw),
        "output_characters": len(output),
        "authoritative_output_tokens": len(token_ids),
        "token_accounting": "completed_response_retokenization",
        "time_to_first_output_seconds": total,
        "atomic_output": True,
        "total_seconds": total,
        "bytes_per_second": len(raw) / total,
        "characters_per_second": len(output) / total,
        "functional_pass": evaluate_functional(output, row["functional_evaluator"]),
        "pointer": pointer,
    }


def _ordinary_request(host: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    state = host.prefill(str(row["prompt"]))
    first = None
    for _ in range(int(row["max_new_tokens"])):
        token = host.decode_step(state)
        if token is None:
            break
        if first is None:
            first = time.perf_counter() - started
    output = host.realize(state).decode("utf-8", errors="strict")
    total = time.perf_counter() - started
    raw = output.encode("utf-8")
    token_ids = host.model_tokenizer.encode(output)
    return {
        "probe_id": row["probe_id"],
        "output": output,
        "output_token_ids": token_ids,
        "output_utf8_bytes": len(raw),
        "output_characters": len(output),
        "authoritative_output_tokens": len(token_ids),
        "token_accounting": "completed_response_retokenization",
        "time_to_first_output_seconds": float(first if first is not None else total),
        "total_seconds": total,
        "bytes_per_second": len(raw) / total,
        "characters_per_second": len(output) / total,
    }


def _qwen_probe(row: Mapping[str, Any]) -> dict[str, Any]:
    return {"probe_id": row["ir_record_id"], "prompt": row["normalized_generation_prompt"], "max_new_tokens": row["generation_max_new_tokens"]}


def _tags_digest(base_url: str, model: str) -> str:
    with urlopen(base_url + "/api/tags", timeout=10) as response:
        rows = json.loads(response.read()).get("models", [])
    matches = [row for row in rows if row.get("name") == model or row.get("model") == model]
    if len(matches) != 1:
        raise Phase3Error("pinned Qwen tag is absent or ambiguous")
    return str(matches[0].get("digest", ""))


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    prospective = _jsonl(root / protocol["prospective_suite"])
    selected, scheduled = select_prospective(prospective, int(protocol["runtime"]["distinct_prompts"]), int(protocol["runtime"]["repeated_observations"]))
    probes = development_probes(root / protocol["development_catalog"])
    ordinary_distinct, ordinary_scheduled = runtime.select_runtime_probes(probes, 100, 20)
    digest = _tags_digest(protocol["transformer_baseline"]["base_url"], protocol["transformer_baseline"]["model"])
    gates = {
        "prospective_depth": len(selected) == 100 and len(scheduled) == 120,
        "ordinary_depth": len(ordinary_distinct) == 100 and len(ordinary_scheduled) == 120,
        "prospective_schedule_bound": hashlib.sha256("\n".join(str(row["ir_record_id"]) for row in scheduled).encode()).hexdigest() == protocol["runtime"]["prospective_schedule_sha256"],
        "ordinary_schedule_bound": hashlib.sha256("\n".join(str(row["probe_id"]) for row in ordinary_scheduled).encode()).hexdigest() == protocol["runtime"]["ordinary_schedule_sha256"],
        "qwen_digest_bound": digest == protocol["transformer_baseline"]["digest"],
        "cuda_initially_unused": int(torch.cuda.memory_allocated()) == 0,
        "training_prohibited": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "status": "PASS_V19_CPU_RUNTIME_PREFLIGHT" if all(gates.values()) else "FAIL_V19_CPU_RUNTIME_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
        "prospective_distinct": len(selected),
        "prospective_observations": len(scheduled),
        "ordinary_distinct": len(ordinary_distinct),
        "ordinary_observations": len(ordinary_scheduled),
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_present": False,
        "final_test_accessed": False,
    }


@torch.inference_mode()
def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v19 runtime output exists: {output}")
    if not preflight(root, protocol_path)["status"].startswith("PASS"):
        raise Phase3Error("v19 runtime preflight failed")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = protocol["runtime"]
    prospective = _jsonl(root / protocol["prospective_suite"])
    distinct, scheduled = select_prospective(prospective, int(cfg["distinct_prompts"]), int(cfg["repeated_observations"]))
    probes = development_probes(root / protocol["development_catalog"])
    ordinary_distinct, ordinary_scheduled = runtime.select_runtime_probes(probes, 100, 20)
    reference = {row["probe_id"]: row for row in _jsonl(root / protocol["ordinary_reference_outputs"])}
    package = root / protocol["package"]
    metadata = _json(root / protocol["package_metadata"])
    public = (root / protocol["public_key"]).read_bytes()
    Host = _host_api((root / protocol["layercake_root"]).resolve())

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    candidate_cuda_before = int(torch.cuda.memory_allocated())
    process = psutil.Process()
    candidate_idle_rss = process.memory_info().rss
    load_started = time.perf_counter()
    temporary_registry = tempfile.TemporaryDirectory(prefix="abi-phase4-v19-runtime-")
    host = Host(Path(temporary_registry.name), trust_store={metadata["public_key"]["key_id"]: public}, device="cpu")
    activation = host.activate(package)
    load_seconds = time.perf_counter() - load_started
    with runtime.PeakMonitor(lambda: process.memory_info().rss) as candidate_monitor:
        request_started = time.perf_counter()
        candidate_cold = _pointer_request(host, distinct[0])
        candidate_cold["single_cold_request"] = True
        candidate_cold["model_load_seconds"] = load_seconds
        candidate_cold["time_to_first_output_from_cold_start_seconds"] = load_seconds + candidate_cold["time_to_first_output_seconds"]
        candidate_cold["total_from_cold_start_seconds"] = load_seconds + (time.perf_counter() - request_started)
        for row in distinct[: int(cfg["warmup_observations"])]:
            _pointer_request(host, row)
        candidate_rows = [_pointer_request(host, row) for row in scheduled]
        for row in ordinary_distinct[: int(cfg["warmup_observations"])]:
            _ordinary_request(host, row)
        ordinary_rows = [_ordinary_request(host, row) for row in ordinary_scheduled]
    candidate_peak_rss_delta = max(0, candidate_monitor.peak - candidate_idle_rss)
    candidate_active_tensor_bytes = sum(_tensor_bytes(module) for module in (host.model, host.router, host.residual))
    candidate_cuda_after = int(torch.cuda.memory_allocated())
    candidate_cuda_peak = int(torch.cuda.max_memory_allocated())
    del host
    temporary_registry.cleanup()

    base_url = protocol["transformer_baseline"]["base_url"]
    model = protocol["transformer_baseline"]["model"]
    runtime._ollama_unload(base_url, model)
    if runner_working_set() != 0:
        raise Phase3Error("llama-server remained resident after unload")
    original_post = runtime._post_json
    device_records: list[dict[str, Any]] = []

    def patched_post(url: str, body: Mapping[str, Any], *, stream: bool = False):
        return original_post(url, force_cpu_body(url, body), stream=stream)

    runtime._post_json = patched_post
    try:
        with runtime.PeakMonitor(runner_working_set) as qwen_monitor:
            qwen_cold = runtime._ollama_request(base_url, model, _qwen_probe(distinct[0]), cfg["keep_alive"])
            qwen_cold["single_cold_request"] = True
            device = _ps_model(base_url, model)
            device_records.append(device)
            for row in distinct[: int(cfg["warmup_observations"])]:
                runtime._ollama_request(base_url, model, _qwen_probe(row), cfg["keep_alive"])
            qwen_rows = []
            for row in scheduled:
                measured = runtime._ollama_request(base_url, model, _qwen_probe(row), cfg["keep_alive"])
                measured["functional_pass"] = evaluate_functional(measured["output"], row["functional_evaluator"])
                qwen_rows.append(measured)
                device_records.append(_ps_model(base_url, model))
        qwen_peak_rss = qwen_monitor.peak
    finally:
        runtime._post_json = original_post
        runtime._ollama_unload(base_url, model)

    candidate_metrics = runtime._metrics(candidate_rows)
    qwen_metrics = runtime._metrics(qwen_rows)
    ordinary_metrics = runtime._metrics(ordinary_rows)
    paired_speed = runtime.paired_ratio_bootstrap(
        [row["bytes_per_second"] for row in candidate_rows],
        [row["bytes_per_second"] for row in qwen_rows],
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["bootstrap_seed"]),
    )
    paired_quality = paired_quality_bootstrap(
        [row["functional_pass"] for row in candidate_rows],
        [row["functional_pass"] for row in qwen_rows],
        int(protocol["statistics"]["bootstrap_replicates"]),
        int(protocol["statistics"]["quality_bootstrap_seed"]),
    )
    ordinary_exact = sum(
        row["output"] == reference[row["probe_id"]]["output"]
        and row["output_token_ids"] == reference[row["probe_id"]]["output_token_ids"]
        for row in ordinary_rows
    )
    throughput_ratio = candidate_metrics["median_bytes_per_second"] / qwen_metrics["median_bytes_per_second"]
    ttft_ratio = candidate_metrics["median_time_to_first_output_seconds"] / qwen_metrics["median_time_to_first_output_seconds"]
    retention = ordinary_metrics["median_bytes_per_second"] / float(protocol["locked_v18_runtime"]["median_bytes_per_second"])
    candidate_quality = sum(row["functional_pass"] for row in candidate_rows)
    qwen_quality = sum(row["functional_pass"] for row in qwen_rows)
    gates_cfg = protocol["gates"]
    gates = {
        "same_signed_package": activation["archive_hash"] == protocol["bindings"][protocol["package"]],
        "payload_preserved": activation["payload_hash"] == protocol["tensor_payload_hash"],
        "prospective_candidate_quality": candidate_quality == len(candidate_rows),
        "quality_noninferior_to_qwen": paired_quality["lower_95"] >= float(gates_cfg["quality_relative_lower_minimum"]),
        "throughput_ratio_at_least_2x": throughput_ratio >= float(gates_cfg["cpu_throughput_ratio_minimum"]),
        "paired_throughput_lower_at_least_2x": paired_speed["lower_95"] >= float(gates_cfg["paired_bootstrap_lower_minimum"]),
        "ordinary_phase2_throughput_retention": retention >= float(gates_cfg["phase2_host_throughput_retention_minimum"]),
        "ordinary_output_identity": ordinary_exact == len(ordinary_rows),
        "ttft_advantage": ttft_ratio <= float(gates_cfg["ttft_ratio_maximum"]),
        "lower_active_tensor_bytes": candidate_active_tensor_bytes < int(protocol["transformer_baseline"]["model_file_bytes"]),
        "lower_peak_active_rss": candidate_peak_rss_delta < qwen_peak_rss,
        "candidate_fully_cpu": candidate_cuda_before == candidate_cuda_after == candidate_cuda_peak == 0,
        "qwen_fully_cpu": len(device_records) == 121 and all(int(row.get("size_vram", -1)) == 0 for row in device_records),
        "genuine_candidate_cold_single_request": candidate_cold["single_cold_request"] and candidate_cold["model_load_seconds"] > 0,
        "genuine_qwen_cold_single_request": qwen_cold["single_cold_request"] and qwen_cold["load_seconds_reported"] > 0,
        "pointer_physical_execution": all(row["pointer"].get("candidate_count") == 6 and row["pointer"].get("candidate_scoring_forward_passes") == 1 and row["pointer"].get("active_residual_routes") == 1 and row["pointer"].get("evaluator_used") is False for row in candidate_rows),
        "authoritative_token_accounting": all(row["token_accounting"] == "completed_response_retokenization" for row in candidate_rows + ordinary_rows) and all(row["authoritative_output_tokens"] >= 0 for row in qwen_rows),
        "depth": len({row["probe_id"] for row in candidate_rows}) == 100 and len(candidate_rows) == len(qwen_rows) == 120,
        "p95_supported": candidate_metrics["p95_supported"] and qwen_metrics["p95_supported"],
        "p99_not_promoted": not candidate_metrics["p99_supported"] and not qwen_metrics["p99_supported"],
        "receiver_learning_zero": activation["receiver_training_steps"] == activation["receiver_calibration_runs"] == 0,
        "teacher_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    output.mkdir(parents=True)
    observations = output / "observations.jsonl"
    _write_immutable(observations, b"".join(canonical_json_bytes({"system": system, "mode": mode, **row}) for system, mode, rows in (("layercake_v19", "prompt_span", candidate_rows), ("qwen", "prompt_span", qwen_rows), ("layercake_v19", "ordinary", ordinary_rows)) for row in rows))
    result = {
        "format": "abi-capability-compiler-phase4-v19-cpu-runtime-result/1",
        "status": "PASS_V19_SAME_ARTIFACT_CPU_RUNTIME" if all(gates.values()) else "FAIL_V19_SAME_ARTIFACT_CPU_RUNTIME",
        "protocol_sha256": protocol_sha,
        "package_sha256": sha256_file(package),
        "tensor_payload_hash": protocol["tensor_payload_hash"],
        "candidate": {**candidate_metrics, "functional_passes": candidate_quality, "active_tensor_bytes": candidate_active_tensor_bytes, "peak_active_rss_delta_bytes": candidate_peak_rss_delta, "cold": candidate_cold, "ordinary": {**ordinary_metrics, "exact_outputs": ordinary_exact, "throughput_retention": retention}},
        "optimized_transformer": {**qwen_metrics, "functional_passes": qwen_quality, "model": model, "digest": protocol["transformer_baseline"]["digest"], "model_file_bytes": protocol["transformer_baseline"]["model_file_bytes"], "peak_runner_rss_bytes": qwen_peak_rss, "cold": qwen_cold},
        "comparisons": {"median_throughput_ratio": throughput_ratio, "paired_throughput": paired_speed, "paired_quality": paired_quality, "median_ttft_ratio": ttft_ratio, "peak_rss_ratio": candidate_peak_rss_delta / qwen_peak_rss},
        "device_control": {"candidate_cuda_allocated_before_bytes": candidate_cuda_before, "candidate_cuda_allocated_after_bytes": candidate_cuda_after, "candidate_cuda_peak_allocated_bytes": candidate_cuda_peak, "qwen_size_vram_zero_observations": sum(int(row.get("size_vram", -1)) == 0 for row in device_records), "qwen_device_observations": len(device_records)},
        "prompt_depth": {"distinct": 100, "repeated": 20, "observations_per_headline_system": 120},
        "hardware": {"machine": platform.node(), "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False), "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads()},
        "gates": gates,
        "raw_observations_sha256": sha256_file(observations),
        "teacher_present": False,
        "training_performed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Same-artifact v19 prompt-span CPU runtime and ordinary-retention gate only; stable imported-information frontier, matched LoRA/distillation, final test, Phase 4, and ABI superiority remain unproven.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
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
    result = preflight(root, root / args.protocol) if args.command == "preflight" else run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
