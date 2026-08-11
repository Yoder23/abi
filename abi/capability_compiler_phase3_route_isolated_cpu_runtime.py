"""Exact-lineage fully CPU runtime certification for route-isolated Phase 3 A0."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import torch

from . import capability_compiler_phase3_cpu_runtime as base
from . import capability_compiler_phase3_cpu_runtime_v2 as qwen_cpu
from . import capability_compiler_phase3_cpu_runtime_v3 as candidate_cpu
from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_route_isolated import RouteIsolatedResidual


FORMAT = "abi-capability-compiler-phase3-route-isolated-fully-cpu-runtime/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def adapt_screen_protocol(screen: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    adapted = copy.deepcopy(dict(screen))
    adapted["candidate"] = {
        "checkpoint": protocol["candidate"]["checkpoint"],
        "checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "immutable": True,
    }
    return adapted


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    repair = protocol.get("route_isolated_runtime", {})
    if (
        protocol.get("format") != base.FORMAT
        or protocol.get("status") != "PREREGISTERED_MATCHED_CPU_RUNTIME_TTFT_RSS_SCREEN"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or repair.get("format") != FORMAT
        or repair.get("candidate_residual") != "RouteIsolatedResidual"
        or repair.get("active_rank") != 16
        or repair.get("physical_experts") != 4
        or repair.get("experts_active_per_token") != 1
        or repair.get("ollama_num_gpu") != 0
        or repair.get("maximum_candidate_cuda_allocated_bytes") != 0
        or int(protocol["runtime"]["distinct_prompts"]) < 100
        or int(protocol["runtime"]["repeated_observations"]) < 20
        or int(protocol["runtime"]["torch_threads"]) != 1
    ):
        raise Phase3Error("route-isolated CPU runtime governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"route-isolated CPU runtime binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable route-isolated CPU runtime output exists")
    original_json = base._json
    original_residual = base.SharedWeakResidual
    original_router_loader = base._load_router
    original_score = sparse._score
    original_post = base._post_json
    original_request = base._ollama_request
    screen_path = (root / protocol["candidate"]["screen_protocol"]).resolve()
    device_records: list[dict[str, Any]] = []

    def patched_json(path: Path) -> dict[str, Any]:
        document = original_json(path)
        return adapt_screen_protocol(document, protocol) if path.resolve() == screen_path else document

    def patched_post(url: str, body: Mapping[str, Any], *, stream: bool = False):
        return original_post(url, qwen_cpu.force_cpu_body(url, body), stream=stream)

    def patched_request(base_url: str, model: str, probe: Mapping[str, Any], keep_alive: str):
        row = original_request(base_url, model, probe, keep_alive)
        device = qwen_cpu._ps_model(base_url, model)
        row["ollama_size_bytes"] = int(device.get("size", 0))
        row["ollama_size_vram_bytes"] = int(device.get("size_vram", -1))
        device_records.append(
            {
                "probe_id": row["probe_id"],
                "size": row["ollama_size_bytes"],
                "size_vram": row["ollama_size_vram_bytes"],
            }
        )
        return row

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = int(torch.cuda.memory_allocated())
    base._json = patched_json
    base.SharedWeakResidual = RouteIsolatedResidual
    base._load_router = candidate_cpu._load_router_cpu
    sparse._score = candidate_cpu._score_on_model_device
    base._post_json = patched_post
    base._ollama_request = patched_request
    try:
        raw_result = base.run(root, protocol_path, output / "raw")
    finally:
        base._json = original_json
        base.SharedWeakResidual = original_residual
        base._load_router = original_router_loader
        sparse._score = original_score
        base._post_json = original_post
        base._ollama_request = original_request
    after = int(torch.cuda.memory_allocated())
    peak = int(torch.cuda.max_memory_allocated())

    raw_rows = [json.loads(line) for line in (output / "raw" / "observations.jsonl").open(encoding="utf-8")]
    qwen_rows = [row for row in raw_rows if row["system"] == "qwen"]
    qwen_fully_cpu = (
        len(qwen_rows) == 120
        and len(device_records) >= 124
        and all(row.get("ollama_size_vram_bytes") == 0 and row.get("ollama_size_bytes", 0) > 0 for row in qwen_rows)
        and raw_result["optimized_transformer"]["cold"].get("ollama_size_vram_bytes") == 0
    )
    candidate_fully_cpu = before == 0 and after == 0 and peak == 0
    checkpoint_bound = sha256_file(root / protocol["candidate"]["checkpoint"]) == protocol["candidate"]["checkpoint_sha256"]
    exact_outputs = int(raw_result["candidate"]["outputs_exact_to_v494"])
    gates = dict(raw_result["gates"])
    gates.update(
        {
            "exact_route_isolated_checkpoint_bound": checkpoint_bound,
            "runtime_outputs_exact_to_route_isolated_quality_candidate": exact_outputs == 120,
            "optimized_transformer_physically_cpu_only": qwen_fully_cpu,
            "candidate_router_features_and_model_physically_cpu_only": candidate_fully_cpu,
        }
    )
    passed = all(gates.values())
    result = {
        "format": FORMAT,
        "status": "PASS_EXACT_ROUTE_ISOLATED_FULLY_CPU_RUNTIME" if passed else "FAIL_EXACT_ROUTE_ISOLATED_FULLY_CPU_RUNTIME_GATE",
        "protocol_sha256": protocol_sha256,
        "raw_result_path": "raw/result.json",
        "raw_result_sha256": sha256_file(output / "raw" / "result.json"),
        "raw_observations_sha256": sha256_file(output / "raw" / "observations.jsonl"),
        "candidate_checkpoint_sha256": protocol["candidate"]["checkpoint_sha256"],
        "candidate": {
            **{key: value for key, value in raw_result["candidate"].items() if key != "outputs_exact_to_v494"},
            "outputs_exact_to_route_isolated_reference": exact_outputs,
        },
        "phase2_host_without_bridge": raw_result["phase2_host_without_bridge"],
        "optimized_transformer": raw_result["optimized_transformer"],
        "comparisons": raw_result["comparisons"],
        "device_control": {
            "candidate_cuda_allocated_before_bytes": before,
            "candidate_cuda_allocated_after_bytes": after,
            "candidate_cuda_peak_allocated_bytes": peak,
            "candidate_fully_cpu": candidate_fully_cpu,
            "ollama_num_gpu": 0,
            "qwen_device_observations": len(device_records),
            "qwen_headline_observations": len(qwen_rows),
            "qwen_all_headline_size_vram_zero": qwen_fully_cpu,
            "qwen_median_loaded_size_bytes": statistics.median(row["ollama_size_bytes"] for row in qwen_rows),
        },
        "prompt_depth": raw_result["prompt_depth"],
        "hardware": raw_result["hardware"],
        "gates": gates,
        "passed": passed,
        "teacher_present_at_inference": False,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
