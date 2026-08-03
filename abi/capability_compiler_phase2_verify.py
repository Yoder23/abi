"""Verify the hash-bound Phase 2 preregistration and, later, its evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase1_certificate import verify_certificate
from .capability_compiler_phase2_common import PHASE1_IR_SHA256, Phase2Error, sha256_file


PROTOCOL = "ABI_CAPABILITY_COMPILER_PHASE2_PROTOCOL_REPAIR1_V2.json"


def verify_protocol(root: Path, protocol_path: Path | None = None) -> dict[str, Any]:
    path = (protocol_path or root / PROTOCOL).resolve()
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("format") != "abi-capability-compiler-phase2-protocol-repair/2":
        raise Phase2Error("invalid Phase 2 repair protocol format")
    if amendment.get("status") != "PREREGISTERED_REPAIR1_BEFORE_ANY_PHASE2_TRAINING":
        raise Phase2Error("Phase 2 repair was not preregistered")
    if (
        amendment.get("candidate_training_performed_before_repair") is not False
        or amendment.get("repair_count") != 1
        or amendment.get("repairs_remaining") != 0
    ):
        raise Phase2Error("repair boundary changed")
    original_path = (root / amendment["original_protocol"]["path"]).resolve()
    if root.resolve() not in original_path.parents or sha256_file(original_path) != amendment["original_protocol"]["sha256"]:
        raise Phase2Error("original Phase 2 protocol changed")
    protocol = json.loads(original_path.read_text(encoding="utf-8"))
    if (
        protocol.get("format") != "abi-capability-compiler-phase2-protocol/1"
        or protocol.get("status") != "PREREGISTERED_BEFORE_ANY_PHASE2_TRAINING"
        or protocol.get("candidate_training_performed_before_preregistration") is not False
    ):
        raise Phase2Error("original Phase 2 protocol was not preregistered")
    failure_path = (root / amendment["failure_evidence"]["path"]).resolve()
    if root.resolve() not in failure_path.parents or sha256_file(failure_path) != amendment["failure_evidence"]["sha256"]:
        raise Phase2Error("repair failure evidence changed")
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if failure.get("phase2_training_started") is not False or failure.get("repair_budget_consumed") != 1:
        raise Phase2Error("repair evidence does not prove the training boundary")
    phase1 = root / "ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json"
    verify_certificate(phase1)
    if sha256_file(phase1) != protocol["phase1"]["certificate_sha256"]:
        raise Phase2Error("Phase 1 certificate binding changed")
    ir = root / protocol["phase1"]["ir_path"]
    if sha256_file(ir) != PHASE1_IR_SHA256 or protocol["phase1"]["ir_sha256"] != PHASE1_IR_SHA256:
        raise Phase2Error("Phase 1 IR binding changed")
    for relative, expected in amendment["implementation_bindings"].items():
        target = (root / relative).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            raise Phase2Error("unsafe or missing implementation binding")
        if sha256_file(target) != expected:
            raise Phase2Error(f"Phase 2 implementation changed: {relative}")
    student = protocol["student"]
    if (
        student["parameter_count"] != 11_060_800
        or student["deployed_bfloat16_bytes"] != 22_121_600
        or student["context_tokens"] != 768
        or student["active_byte_ratio_vs_layercake"] >= 1.02
    ):
        raise Phase2Error("same-size transformer lock changed")
    if protocol["splits"]["final_access"] != "PROHIBITED_DURING_PHASE2":
        raise Phase2Error("Phase 2 final-set firewall changed")
    if protocol["statistics"] != {
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 1729,
        "confidence_level": 0.95,
        "headline_seeds": [104729, 130363, 155921],
        "minimum_prompts_per_capability": 100,
        "minimum_warm_runtime_observations": 20,
        "p95_minimum_observations": 100,
        "p99_minimum_observations": 1000,
        "primary_throughput": ["bytes_per_second", "characters_per_second"],
    }:
        raise Phase2Error("Phase 2 statistical lock changed")
    return {
        "status": "PASS",
        "protocol_path": path.as_posix(),
        "protocol_sha256": sha256_file(path),
        "implementation_bindings": len(amendment["implementation_bindings"]),
        "repair_count": 1,
        "candidate_training_performed": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify_protocol(root, Path(args.protocol).resolve() if args.protocol else None)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
