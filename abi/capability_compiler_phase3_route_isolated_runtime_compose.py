"""Hostile composition of exact route-isolated runtime and corrected Qwen RSS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-route-isolated-runtime-composition/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _verify_embedded(document: Mapping[str, Any]) -> None:
    payload = dict(document)
    expected = payload.pop("evidence_sha256", None)
    if expected != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise Phase3Error("embedded evidence hash changed")


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CORRECTED_RSS_GATE_COMPOSITION"
        or protocol.get("new_generation_authorized") is not False
        or protocol.get("performance_rerun_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("route runtime composition governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"route runtime composition binding changed: {relative}")
    return protocol, sha256_file(path)


def compose(runtime: Mapping[str, Any], rss: Mapping[str, Any]) -> dict[str, Any]:
    device = runtime.get("device_control", {})
    if device.get("candidate_fully_cpu") is not True or any(
        int(device.get(name, -1)) != 0
        for name in (
            "candidate_cuda_allocated_before_bytes",
            "candidate_cuda_allocated_after_bytes",
            "candidate_cuda_peak_allocated_bytes",
        )
    ):
        raise Phase3Error("candidate device evidence changed")
    if device.get("qwen_all_headline_size_vram_zero") is not True:
        raise Phase3Error("runtime Qwen device evidence changed")
    if rss.get("qwen", {}).get("processor") != "100% CPU" or int(rss["qwen"].get("size_vram_bytes", -1)) != 0:
        raise Phase3Error("corrected Qwen device evidence changed")
    if runtime.get("optimized_transformer", {}).get("digest") != rss.get("qwen", {}).get("digest"):
        raise Phase3Error("Qwen digest differs across evidence")
    gates = dict(runtime.get("gates", {}))
    if [name for name, value in gates.items() if not value] != ["lower_peak_active_rss"]:
        raise Phase3Error("exact runtime failure boundary changed")
    candidate_peak = int(runtime["candidate"]["peak_active_rss_delta_bytes"])
    if int(rss.get("candidate", {}).get("sealed_peak_process_rss_delta_bytes", -1)) != candidate_peak:
        raise Phase3Error("sealed candidate RSS differs across evidence")
    qwen_peak = int(rss["qwen"]["monitored_peak_runner_working_set_bytes"])
    corrected = candidate_peak < qwen_peak
    gates["lower_peak_active_rss"] = corrected
    if not corrected or not all(gates.values()) or rss.get("passed") is not True:
        raise Phase3Error("corrected route runtime gates do not all pass")
    return {
        "candidate_peak_rss_delta_bytes": candidate_peak,
        "qwen_runner_peak_working_set_bytes": qwen_peak,
        "candidate_to_qwen_peak_rss_ratio": candidate_peak / qwen_peak,
        "corrected_gates": gates,
    }


def _must_reject(name: str, callback: Any) -> str:
    try:
        callback()
    except Phase3Error:
        return name
    raise Phase3Error(f"route runtime composition accepted hostile mutation: {name}")


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable route runtime composition output exists")
    runtime = _json(root / protocol["evidence"]["fully_cpu_runtime"])
    rss = _json(root / protocol["evidence"]["qwen_rss"])
    _verify_embedded(runtime)
    _verify_embedded(rss)
    composed = compose(runtime, rss)
    rejected = []
    high_candidate = copy.deepcopy(runtime)
    high_candidate["candidate"]["peak_active_rss_delta_bytes"] = composed["qwen_runner_peak_working_set_bytes"] + 1
    high_rss = copy.deepcopy(rss)
    high_rss["candidate"]["sealed_peak_process_rss_delta_bytes"] = composed["qwen_runner_peak_working_set_bytes"] + 1
    rejected.append(_must_reject("candidate_rss_regression", lambda: compose(high_candidate, high_rss)))
    bad_digest = copy.deepcopy(rss)
    bad_digest["qwen"]["digest"] = "changed"
    rejected.append(_must_reject("qwen_digest_mutation", lambda: compose(runtime, bad_digest)))
    bad_device = copy.deepcopy(runtime)
    bad_device["device_control"]["candidate_cuda_peak_allocated_bytes"] = 1
    rejected.append(_must_reject("candidate_device_mutation", lambda: compose(bad_device, rss)))
    bad_gate = copy.deepcopy(runtime)
    bad_gate["gates"]["qualified_router_exact"] = False
    rejected.append(_must_reject("non_rss_gate_mutation", lambda: compose(bad_gate, rss)))
    bad_binding = copy.deepcopy(rss)
    bad_binding["candidate"]["sealed_peak_process_rss_delta_bytes"] += 1
    rejected.append(_must_reject("candidate_rss_binding_mutation", lambda: compose(runtime, bad_binding)))
    result = {
        "format": FORMAT,
        "status": "PASS_EXACT_ROUTE_ISOLATED_CORRECTED_FULLY_CPU_RUNTIME_GATE_MATRIX",
        "protocol_sha256": protocol_sha256,
        **composed,
        "candidate_checkpoint_sha256": runtime["candidate_checkpoint_sha256"],
        "candidate_median_bytes_per_second": runtime["candidate"]["median_bytes_per_second"],
        "qwen_median_bytes_per_second": runtime["optimized_transformer"]["median_bytes_per_second"],
        "candidate_to_qwen_median_throughput_ratio": runtime["comparisons"]["candidate_to_qwen_median_bytes_per_second_ratio"],
        "paired_throughput_lower_95": runtime["comparisons"]["paired_throughput"]["lower_95"],
        "candidate_to_qwen_median_ttft_ratio": runtime["comparisons"]["candidate_to_qwen_median_ttft_ratio"],
        "candidate_to_parent_throughput_retention": runtime["comparisons"]["candidate_to_parent_median_throughput_retention"],
        "runtime_outputs_exact_to_reference": runtime["candidate"]["outputs_exact_to_route_isolated_reference"],
        "hostile_mutations_rejected": rejected,
        "hostile_mutations_rejected_count": len(rejected),
        "new_generation_performed": False,
        "performance_rerun_performed": False,
        "historical_evidence_changed": False,
        "promotion_eligible": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.mkdir(parents=True)
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
