"""Device-control repair for the V503 matched CPU runtime screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping
from urllib.request import urlopen

from . import capability_compiler_phase3_cpu_runtime as base
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-matched-cpu-runtime-device-repair/1"


def force_cpu_body(url: str, body: Mapping[str, Any]) -> dict[str, Any]:
    repaired = dict(body)
    if url.endswith("/api/chat"):
        options = dict(repaired.get("options", {})); options["num_gpu"] = 0; repaired["options"] = options
    return repaired


def _ps_model(base_url: str, model: str) -> dict[str, Any]:
    with urlopen(base_url + "/api/ps", timeout=10) as response: rows = json.loads(response.read()).get("models", [])
    matches = [row for row in rows if row.get("name") == model or row.get("model") == model]
    if len(matches) != 1: raise Phase3Error("Ollama CPU device record absent or ambiguous")
    return matches[0]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != base.FORMAT or protocol.get("status") != "PREREGISTERED_MATCHED_CPU_RUNTIME_TTFT_RSS_SCREEN" or protocol.get("runtime_repair", {}).get("format") != FORMAT or protocol["runtime_repair"].get("ollama_num_gpu") != 0 or protocol["runtime_repair"].get("required_size_vram_bytes") != 0: raise Phase3Error("CPU device repair governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"CPU device repair binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable repaired CPU runtime output exists: {output}")
    original_post = base._post_json; original_request = base._ollama_request; device_records: list[dict[str, Any]] = []
    def patched_post(url: str, body: Mapping[str, Any], *, stream: bool = False): return original_post(url, force_cpu_body(url, body), stream=stream)
    def patched_request(base_url: str, model: str, probe: Mapping[str, Any], keep_alive: str):
        row = original_request(base_url, model, probe, keep_alive); device = _ps_model(base_url, model); row["ollama_size_bytes"] = int(device.get("size", 0)); row["ollama_size_vram_bytes"] = int(device.get("size_vram", -1)); device_records.append({"probe_id": row["probe_id"], "size": row["ollama_size_bytes"], "size_vram": row["ollama_size_vram_bytes"]}); return row
    base._post_json = patched_post; base._ollama_request = patched_request
    try: base_result = base.run(root, protocol_path, output / "raw_v506")
    finally: base._post_json = original_post; base._ollama_request = original_request
    raw_rows = [json.loads(line) for line in (output / "raw_v506" / "observations.jsonl").open(encoding="utf-8")]; qwen_rows = [row for row in raw_rows if row["system"] == "qwen"]
    cpu_device = len(qwen_rows) == 120 and all(row.get("ollama_size_vram_bytes") == 0 and row.get("ollama_size_bytes", 0) > 0 for row in qwen_rows) and base_result["optimized_transformer"]["cold"].get("ollama_size_vram_bytes") == 0
    gates = dict(base_result["gates"]); gates["optimized_transformer_physically_cpu_only"] = cpu_device; passed = all(gates.values())
    result = {"format": FORMAT, "status": "PASS_VALID_CPU_RUNTIME_PHASE3_CERTIFICATE_REVIEW_OPEN" if passed else "FAIL_VALID_CPU_RUNTIME_GATE_CLOSED", "protocol_sha256": protocol_sha, "base_result_path": "raw_v506/result.json", "base_result_sha256": sha256_file(output / "raw_v506" / "result.json"), "raw_observations_sha256": sha256_file(output / "raw_v506" / "observations.jsonl"), "device_control": {"ollama_num_gpu": 0, "device_observations": len(device_records), "recorded_qwen_headline_observations": len(qwen_rows), "all_recorded_size_vram_zero": cpu_device, "median_loaded_size_bytes": statistics.median(row["ollama_size_bytes"] for row in qwen_rows)}, "candidate": base_result["candidate"], "phase2_host_without_bridge": base_result["phase2_host_without_bridge"], "optimized_transformer": base_result["optimized_transformer"], "comparisons": base_result["comparisons"], "prompt_depth": base_result["prompt_depth"], "hardware": base_result["hardware"], "gates": gates, "passed": passed, "teacher_present_at_inference": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CPU_RUNTIME_REPAIR_PROTOCOL_V505.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_cpu_runtime/runtime_v506"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
