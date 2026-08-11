"""Corrected repeated llama-server RSS attribution for the V506 CPU screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Iterable
from urllib.request import urlopen

import psutil

from . import capability_compiler_phase3_cpu_runtime as runtime
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_cpu_runtime_v2 import force_cpu_body


FORMAT = "abi-capability-compiler-phase3-qwen-runner-rss-audit/1"


def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_REPEATED_QWEN_RUNNER_RSS_ATTRIBUTION" or protocol.get("neural_training_authorized") is not False or protocol.get("performance_rerun_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("Qwen RSS audit governance changed")
    if int(protocol["rss_audit"]["repeated_observations"]) < 20 or protocol["rss_audit"]["runner_process_name"] != "llama-server": raise Phase3Error("Qwen RSS audit depth or process identity changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"Qwen RSS audit binding changed: {relative}")
    return protocol, sha256_file(path)


def _runner_processes() -> list[psutil.Process]:
    output = []
    for process in psutil.process_iter(["name"]):
        try:
            if str(process.info["name"] or "").casefold() == "llama-server.exe" or str(process.info["name"] or "").casefold() == "llama-server": output.append(process)
        except (psutil.Error, OSError): pass
    return output


def runner_working_set() -> int:
    total = 0
    for process in _runner_processes():
        try: total += int(process.memory_info().rss)
        except (psutil.Error, OSError): pass
    return total


def runner_private_bytes() -> int:
    total = 0
    for process in _runner_processes():
        try: total += int(process.memory_info().private)
        except (psutil.Error, OSError, AttributeError): pass
    return total


def _ps_record(base_url: str, model: str) -> dict[str, Any]:
    with urlopen(base_url + "/api/ps", timeout=10) as response: rows = json.loads(response.read()).get("models", [])
    matches = [row for row in rows if row.get("name") == model or row.get("model") == model]
    if len(matches) != 1: raise Phase3Error("Qwen /api/ps record absent or ambiguous")
    return matches[0]


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable Qwen RSS audit output exists: {output}")
    base_url = str(protocol["transformer_baseline"]["base_url"]); model = str(protocol["transformer_baseline"]["model"]); runtime._ollama_unload(base_url, model)
    if runner_working_set() != 0: raise Phase3Error("llama-server remained resident after unload")
    probes = development_probes(root / protocol["development"]["catalog"]); probe = next(row for row in probes if row["probe_id"] == protocol["rss_audit"]["probe_id"])
    original_post = runtime._post_json
    def patched_post(url: str, body: dict[str, Any], *, stream: bool = False): return original_post(url, force_cpu_body(url, body), stream=stream)
    runtime._post_json = patched_post
    try:
        runtime._ollama_request(base_url, model, probe, str(protocol["rss_audit"]["keep_alive"]))
        device = _ps_record(base_url, model)
        if int(device.get("size_vram", -1)) != 0: raise Phase3Error("Qwen RSS audit offloaded to accelerator")
        observations = []
        with runtime.PeakMonitor(runner_working_set) as monitor:
            for index in range(int(protocol["rss_audit"]["repeated_observations"])):
                request = runtime._ollama_request(base_url, model, probe, str(protocol["rss_audit"]["keep_alive"])); observations.append({"index": index, "runner_working_set_bytes": runner_working_set(), "runner_private_bytes": runner_private_bytes(), "request_output_bytes": request["output_utf8_bytes"]})
        peak = monitor.peak
    finally:
        runtime._post_json = original_post; runtime._ollama_unload(base_url, model)
    candidate_peak = int(protocol["candidate"]["sealed_peak_process_rss_delta_bytes"]); ratio = candidate_peak / peak; gate = candidate_peak < peak; within_ten = ratio <= 1.1
    result = {"format": FORMAT, "status": "PASS_CORRECTED_RSS_ATTRIBUTION_PHASE3_REVIEW_OPEN" if gate else "FAIL_CORRECTED_RSS_ATTRIBUTION", "protocol_sha256": protocol_sha, "qwen": {"model": model, "digest": protocol["transformer_baseline"]["digest"], "processor": "100% CPU", "size_vram_bytes": 0, "loaded_size_bytes": int(device["size"]), "runner_process_name": "llama-server", "repeated_observations": len(observations), "median_runner_working_set_bytes": statistics.median(row["runner_working_set_bytes"] for row in observations), "maximum_runner_working_set_bytes": max(row["runner_working_set_bytes"] for row in observations), "monitored_peak_runner_working_set_bytes": peak, "median_runner_private_bytes": statistics.median(row["runner_private_bytes"] for row in observations)}, "candidate": {"sealed_peak_process_rss_delta_bytes": candidate_peak}, "comparison": {"candidate_to_qwen_peak_rss_ratio": ratio, "candidate_lower_peak_rss": gate, "failure_within_ten_percent": (not gate) and within_ten}, "gates": {"physical_cpu_only": True, "repeated_observation_depth": len(observations) >= 20, "runner_physically_observed": peak > 0, "candidate_lower_peak_rss": gate, "final_test_not_accessed": True}, "passed": gate, "optimization_attempt_authorized": (not gate) and within_ten, "observations": observations, "performance_metrics_reused_for_claim": False, "neural_training_performed": False, "historical_evidence_changed": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "v506_arithmetic_erratum": {"recorded_ratio": 1.010355712910959, "correct_ratio": 1.0103557112921597, "effect_on_decision": "none"}}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); output.mkdir(parents=True); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_QWEN_RSS_AUDIT_PROTOCOL_V507.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_qwen_rss_audit/audit_v508"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
