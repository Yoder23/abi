"""Physical device audit of the V506 qualified router path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from . import capability_compiler_phase3_sparse_router as sparse
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_targeted_recovery_bridge import _load_router


FORMAT = "abi-capability-compiler-phase3-candidate-device-audit/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_PHYSICAL_ROUTER_DEVICE_AUDIT" or protocol.get("neural_training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED": raise Phase3Error("candidate device audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"candidate device audit binding changed: {relative}")
    return protocol, sha256_file(path)


def device_set(module: torch.nn.Module) -> list[str]: return sorted({str(value.device) for value in module.parameters()})


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable candidate device audit output exists: {output}")
    screen = json.loads((root / protocol["candidate"]["screen_protocol"]).read_text(encoding="utf-8")); torch.cuda.reset_peak_memory_stats(); router, tokenizer, router_protocol = _load_router(root, screen); before = int(torch.cuda.memory_allocated()); probe = development_probes(root / protocol["development"]["catalog"])[0]; routed, details = sparse._route(router, tokenizer, router_protocol, str(probe["prompt"])); after = int(torch.cuda.memory_allocated()); devices = device_set(router); hybrid = devices != ["cpu"] or before > 0 or after > 0
    result = {"format": FORMAT, "status": "FAIL_CONFIRMED_CANDIDATE_ROUTER_CUDA_DEVICE" if hybrid else "PASS_ROUTER_CPU_ONLY", "protocol_sha256": protocol_sha, "router_parameter_devices": devices, "cuda_allocated_before_route_bytes": before, "cuda_allocated_after_route_bytes": after, "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "probe_id": probe["probe_id"], "expected_route": probe["canonical_capability"], "observed_route": routed, "route_correct": routed == probe["canonical_capability"], "router_segment_count": len(details), "fully_cpu_candidate_path": not hybrid, "neural_training_performed": False, "performance_measured": False, "historical_evidence_changed": False, "promotion_eligible": False, "phase3_certified": False, "final_test_accessed": False}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); output.mkdir(parents=True); _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CANDIDATE_DEVICE_AUDIT_PROTOCOL_V509.json"); parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_candidate_device_audit/audit_v510"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output_dir); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
