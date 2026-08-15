"""Paired-seed replication wrapper for the sealed B40 clarification route."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import capability_compiler_phase4_clarification_route as base
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-clarification-route-replication/1"
SEEDS = (104729, 130363)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_B40_PAIRED_SEED_CLARIFICATION_ROUTE_REPLICATION"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
    ):
        raise Phase3Error("clarification-route replication governance changed")
    if tuple(int(row["seed"]) for row in protocol["runs"]) != SEEDS or any(row["budget"] != "B40" for row in protocol["runs"]):
        raise Phase3Error("clarification-route replication matrix changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"clarification-route replication binding changed: {relative}")
    reference = _json(root / protocol["reference_protocol"])
    for field in ("architecture", "acquisition", "training", "critical_capabilities", "thresholds"):
        if protocol[field] != reference[field]:
            raise Phase3Error(f"clarification-route replication changed sealed field: {field}")
    lineage_protocol = _json(root / protocol["lineage_protocol"])
    return protocol, sha256_file(path), lineage_protocol


def _run(protocol: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    match = next((row for row in protocol["runs"] if int(row["seed"]) == int(seed)), None)
    if match is None:
        raise Phase3Error("unregistered clarification-route replication seed")
    return match


def _runtime_protocol(protocol: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(protocol)
    value["status"] = "PREREGISTERED_B40_HARD_SEED_CLARIFICATION_ROUTE"
    value["run"] = {"budget": "B40", "seed": int(run["seed"])}
    value["lineage_dir"] = str(run["lineage_dir"])
    value["historical_outputs"] = str(run["historical_outputs"])
    value["statistics"] = dict(run["statistics"])
    return value


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    checks = []
    original = base.load_protocol
    try:
        for run in protocol["runs"]:
            runtime = _runtime_protocol(protocol, run)
            base.load_protocol = lambda _root, _path, runtime=runtime: (runtime, protocol_sha, lineage_protocol)
            check = base.preflight(root, protocol_path)
            checks.append({"seed": int(run["seed"]), **check})
    finally:
        base.load_protocol = original
    gates = {
        "two_registered_seeds": len(checks) == 2,
        "both_preflights_pass": all(row["status"] == "PASS_CLARIFICATION_ROUTE_PREFLIGHT" for row in checks),
        "same_selected_information": len({(row["unique_source_attempts"], row["authoritative_teacher_output_tokens"]) for row in checks}) == 1,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-clarification-route-replication-preflight/1",
        "status": "PASS_B40_CLARIFICATION_ROUTE_REPLICATION_PREFLIGHT" if all(gates.values()) else "FAIL_B40_CLARIFICATION_ROUTE_REPLICATION_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "checks": checks,
        "gates": gates,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, seed: int, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    run = _run(protocol, seed)
    runtime = _runtime_protocol(protocol, run)
    original = base.load_protocol
    try:
        base.load_protocol = lambda _root, _path: (runtime, protocol_sha, lineage_protocol)
        trained = base.train(root, protocol_path, output)
    finally:
        base.load_protocol = original
    receipt = {
        "format": "abi-capability-compiler-phase4-clarification-route-replication-training/1",
        "status": "TRAINED_B40_PAIRED_SEED_CLARIFICATION_ROUTE",
        "protocol_sha256": protocol_sha,
        "budget": "B40",
        "seed": int(seed),
        "candidate_metadata_sha256": sha256_file(output / "metadata.json"),
        "checkpoint_sha256": trained["checkpoint"]["sha256"],
        "training": trained["training"],
        "imported_information": trained["imported_information"],
        "teacher_present": False,
        "final_test_accessed": False,
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_immutable(output / "replication_receipt.json", json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
    return receipt


def evaluate(root: Path, protocol_path: Path, seed: int, candidate: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha, lineage_protocol = load_protocol(root, protocol_path)
    run = _run(protocol, seed)
    runtime = _runtime_protocol(protocol, run)
    original = base.load_protocol
    try:
        base.load_protocol = lambda _root, _path: (runtime, protocol_sha, lineage_protocol)
        evaluated = base.evaluate(root, protocol_path, candidate, output)
    finally:
        base.load_protocol = original
    result = {
        "format": "abi-capability-compiler-phase4-clarification-route-replication-result/1",
        "status": "PASS_B40_PAIRED_SEED_CLARIFICATION_ROUTE_MACHINE_GATES" if all(evaluated["gates"].values()) else "FAIL_B40_PAIRED_SEED_CLARIFICATION_ROUTE_MACHINE_GATES",
        "protocol_sha256": protocol_sha,
        "budget": "B40",
        "seed": int(seed),
        "checkpoint_sha256": evaluated["checkpoint_sha256"],
        "underlying_result_sha256": sha256_file(output / "result.json"),
        "raw_outputs_sha256": evaluated["raw_outputs_sha256"],
        "functional_passes_v1": evaluated["functional_passes_v1"],
        "historical_clarification_passes": evaluated["historical_clarification_passes"],
        "candidate_clarification_passes": evaluated["candidate_clarification_passes"],
        "clarification_wilson_lower_95": evaluated["per_capability"]["clarification"]["wilson_v1"]["lower_95"],
        "repetition_collapses_v2": evaluated["repetition_collapses_v2"],
        "teacher_comparison_v1": evaluated["teacher_comparison_v1"],
        "gates": evaluated["gates"],
        "teacher_present_at_inference": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One paired-seed B40 clarification-route development result. No stable minimum, product runtime, final test, Phase 4 certificate, or ABI-superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output / "replication_result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train")
    parser.add_argument("--evaluate")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = root / args.protocol
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.seed is not None and args.train:
        result = train(root, protocol_path, args.seed, root / args.train)
    elif args.seed is not None and args.evaluate and args.output:
        result = evaluate(root, protocol_path, args.seed, root / args.evaluate, root / args.output)
    else:
        raise Phase3Error("select preflight, train, or evaluate with a registered seed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith(("PASS", "TRAINED")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
