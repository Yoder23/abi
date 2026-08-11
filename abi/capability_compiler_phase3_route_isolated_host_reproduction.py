"""Fresh-host reproduction for the exact route-isolated Phase 3 checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import capability_compiler_phase3_final_controls as controls
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_route_isolated import PARAMETERS, SYSTEM, RouteIsolatedResidual


FORMAT = "abi-capability-compiler-phase3-route-isolated-host-reproduction/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, tuple[dict[str, Any], dict[str, Any]]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_TWO_FRESH_HOST_REPRODUCTIONS"
        or tuple(protocol.get("hosts", ())) != ("H1", "H2")
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("route-isolated host reproduction governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"host reproduction binding changed: {relative}")
    control_protocol, _, base = controls.load_protocol(root, root / protocol["base_control_protocol"])
    return protocol, sha256_file(path), (copy.deepcopy(control_protocol), base)


def evaluate_host(root: Path, protocol_path: Path, host: str, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256, bundle = load_protocol(root, protocol_path)
    if host not in protocol["hosts"] or str(output.relative_to(root)).replace("\\", "/") != protocol["host_outputs"][host]:
        raise Phase3Error("host output path changed")
    candidate = root / protocol["candidate_dir"]
    metadata_path = (candidate / "metadata.json").resolve()
    control_protocol, base = bundle
    original = (controls.SharedWeakResidual, controls.EXPECTED_PARAMETERS, controls.SYSTEMS, controls.load_protocol, controls._json)

    def patched_json(path: Path) -> dict[str, Any]:
        document = original[4](path)
        if path.resolve() == metadata_path:
            document = copy.deepcopy(document)
            document["protocol_sha256"] = protocol_sha256
        return document

    controls.SharedWeakResidual = RouteIsolatedResidual
    controls.EXPECTED_PARAMETERS = PARAMETERS
    controls.SYSTEMS = (SYSTEM,)
    controls.load_protocol = lambda _root, _path: (control_protocol, protocol_sha256, base)
    controls._json = patched_json
    try:
        result = controls.evaluate(root, protocol_path, SYSTEM, candidate, output)
    finally:
        controls.SharedWeakResidual, controls.EXPECTED_PARAMETERS, controls.SYSTEMS, controls.load_protocol, controls._json = original
    return {**result, "host_initialization": host}


def assess_hosts(reference_sha256: str, checkpoint_sha256: str, hosts: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    return {
        "two_fresh_hosts_present": len(hosts) == 2,
        "same_unchanged_checkpoint": len(hosts) == 2 and all(row.get("checkpoint_sha256") == checkpoint_sha256 for row in hosts),
        "byte_identical_outputs_to_reference": len(hosts) == 2 and all(row.get("raw_outputs_sha256") == reference_sha256 for row in hosts),
        "quality_exact": len(hosts) == 2 and all(int(row.get("functional_passes_v1", -1)) == 1393 for row in hosts),
        "zero_collapse": len(hosts) == 2 and all(int(row.get("repetition_collapses_v2", -1)) == 0 for row in hosts),
        "router_exact": len(hosts) == 2 and all(int(row.get("router_correct", -1)) == 1400 for row in hosts),
        "strong_routes_exact": len(hosts) == 2 and all(int(row.get("strong_routes_exact", -1)) == 1000 for row in hosts),
        "complete_depth": len(hosts) == 2 and all(int(row.get("observations", -1)) == 1400 for row in hosts),
        "final_test_not_accessed": len(hosts) == 2 and all(row.get("final_test_accessed") is False for row in hosts),
    }


def _must_reject(name: str, gates: Mapping[str, bool]) -> str:
    if not all(gates.values()):
        return name
    raise Phase3Error(f"host reproduction accepted hostile mutation: {name}")


def decide(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256, _ = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable host reproduction decision exists")
    reference = root / protocol["reference_outputs"]
    reference_sha256 = sha256_file(reference)
    if reference_sha256 != protocol["reference_outputs_sha256"]:
        raise Phase3Error("reference outputs changed")
    hosts = []
    for host in protocol["hosts"]:
        directory = root / protocol["host_outputs"][host]
        result = _json(directory / "result.json")
        if result.get("protocol_sha256") != protocol_sha256 or sha256_file(directory / "development_outputs.jsonl") != result.get("raw_outputs_sha256"):
            raise Phase3Error("fresh host evidence binding changed")
        hosts.append(result)
    gates = assess_hosts(reference_sha256, protocol["checkpoint_sha256"], hosts)
    rejected = []
    mutations = []
    for field, value in (
        ("checkpoint_sha256", "changed"),
        ("raw_outputs_sha256", "changed"),
        ("functional_passes_v1", 1392),
        ("repetition_collapses_v2", 1),
        ("router_correct", 1399),
    ):
        mutation = copy.deepcopy(hosts)
        mutation[0][field] = value
        mutations.append((field, mutation))
    mutations.append(("missing_host", hosts[:1]))
    for name, mutation in mutations:
        rejected.append(_must_reject(name, assess_hosts(reference_sha256, protocol["checkpoint_sha256"], mutation)))
    passed = all(gates.values()) and len(rejected) == 6
    result = {
        "format": FORMAT,
        "status": "PASS_THREE_HOST_EXACT_ROUTE_ISOLATED_REPRODUCTION" if passed else "FAIL_ROUTE_ISOLATED_HOST_REPRODUCTION",
        "protocol_sha256": protocol_sha256,
        "reference_host": {"outputs_sha256": reference_sha256, "checkpoint_sha256": protocol["checkpoint_sha256"]},
        "fresh_hosts": [
            {"host": host, "checkpoint_sha256": row["checkpoint_sha256"], "outputs_sha256": row["raw_outputs_sha256"], "functional_passes_v1": row["functional_passes_v1"], "repetition_collapses_v2": row["repetition_collapses_v2"], "evaluation_evidence_sha256": row["evidence_sha256"]}
            for host, row in zip(protocol["hosts"], hosts)
        ],
        "host_initializations": 3,
        "gates": gates,
        "hostile_mutations_rejected": rejected,
        "hostile_mutations_rejected_count": len(rejected),
        "passed": passed,
        "teacher_present_at_inference": False,
        "historical_evidence_changed": False,
        "phase3_certified": False,
        "final_test_accessed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate_parser = sub.add_parser("evaluate-host")
    evaluate_parser.add_argument("--host", choices=("H1", "H2"), required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    decide_parser = sub.add_parser("decide")
    decide_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = evaluate_host(root, root / args.protocol, args.host, root / args.output_dir) if args.command == "evaluate-host" else decide(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
