"""Hostile composition of fully CPU V512 runtime and corrected V508 Qwen RSS."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase3-corrected-runtime-evidence-composition/1"


def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_CORRECTED_RSS_GATE_COMPOSITION" or protocol.get("new_generation_authorized") is not False or protocol.get("performance_rerun_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("runtime evidence composition governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"runtime evidence composition binding changed: {relative}")
    return protocol, sha256_file(path)


def compose(runtime: Mapping[str, Any], rss: Mapping[str, Any]) -> dict[str, Any]:
    if runtime.get("candidate_device", {}).get("fully_cpu") is not True or any(int(runtime["candidate_device"].get(name, -1)) != 0 for name in ("cuda_allocated_before_bytes", "cuda_allocated_after_bytes", "cuda_peak_allocated_bytes")): raise Phase3Error("candidate device evidence changed")
    if rss.get("qwen", {}).get("processor") != "100% CPU" or int(rss["qwen"].get("size_vram_bytes", -1)) != 0: raise Phase3Error("Qwen device evidence changed")
    if runtime.get("optimized_transformer", {}).get("digest") != rss.get("qwen", {}).get("digest"): raise Phase3Error("Qwen digest differs across runtime and RSS evidence")
    gates = dict(runtime.get("gates", {})); failures = [name for name, value in gates.items() if not value]
    if failures != ["lower_peak_active_rss"]: raise Phase3Error("V512 failure boundary changed")
    candidate_peak = int(runtime["candidate"]["peak_active_rss_delta_bytes"]); qwen_peak = int(rss["qwen"]["monitored_peak_runner_working_set_bytes"]); ratio = candidate_peak / qwen_peak; corrected = candidate_peak < qwen_peak; gates["lower_peak_active_rss"] = corrected
    if not corrected or not all(gates.values()): raise Phase3Error("corrected runtime gates do not all pass")
    return {"candidate_peak_rss_delta_bytes": candidate_peak, "qwen_runner_peak_working_set_bytes": qwen_peak, "candidate_to_qwen_peak_rss_ratio": ratio, "corrected_gates": gates}


def _must_reject(name: str, callback: Any) -> str:
    try: callback()
    except Phase3Error: return name
    raise Phase3Error(f"runtime composition accepted hostile mutation: {name}")


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable runtime composition output exists: {output}")
    runtime = _json(root / protocol["evidence"]["fully_cpu_runtime"]); rss = _json(root / protocol["evidence"]["qwen_rss"]); composed = compose(runtime, rss); rejected = []
    high_candidate = copy.deepcopy(runtime); high_candidate["candidate"]["peak_active_rss_delta_bytes"] = int(rss["qwen"]["monitored_peak_runner_working_set_bytes"]) + 1
    rejected.append(_must_reject("candidate_rss_regression", lambda: compose(high_candidate, rss)))
    bad_digest = copy.deepcopy(rss); bad_digest["qwen"]["digest"] = "changed"
    rejected.append(_must_reject("qwen_digest_mutation", lambda: compose(runtime, bad_digest)))
    bad_device = copy.deepcopy(runtime); bad_device["candidate_device"]["cuda_peak_allocated_bytes"] = 1
    rejected.append(_must_reject("candidate_device_mutation", lambda: compose(bad_device, rss)))
    bad_gate = copy.deepcopy(runtime); bad_gate["gates"]["qualified_router_exact"] = False
    rejected.append(_must_reject("non_rss_gate_mutation", lambda: compose(bad_gate, rss)))
    result = {"format": FORMAT, "status": "PASS_CORRECTED_FULLY_CPU_RUNTIME_GATE_MATRIX_PHASE3_REVIEW_OPEN", "protocol_sha256": protocol_sha, **composed, "hostile_mutations_rejected": rejected, "hostile_mutations_rejected_count": len(rejected), "new_generation_performed": False, "performance_rerun_performed": False, "historical_evidence_changed": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False, "v512_arithmetic_erratum": {"recorded_diagnostic_ratio": 0.7142653490675001, "correct_ratio": composed["candidate_to_qwen_peak_rss_ratio"], "effect_on_decision": "none"}}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); output.mkdir(parents=True); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_RUNTIME_EVIDENCE_COMPOSE_PROTOCOL_V513.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_runtime_evidence_compose/composition_v514"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
