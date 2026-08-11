"""Read-only Phase 3 certificate-readiness audit.

This module deliberately distinguishes endpoint qualification from causal-control
qualification.  Evidence from an early matched A0--A4 experiment cannot be
spliced onto a later, materially different endpoint whose absolute quality
passes only after additional acquisition stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
)


FORMAT = "abi-capability-compiler-phase3-certificate-readiness/1"
EXPECTED_CONTROLS = ("parent", "A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        probe_id = str(row["probe_id"])
        if probe_id in result:
            raise Phase3Error(f"duplicate probe: {probe_id}")
        result[probe_id] = row
    return result


def validate_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CERTIFICATE_READINESS_AUDIT"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("historical_evidence_mutation") != "PROHIBITED"
        or tuple(protocol.get("required_final_controls", ())) != EXPECTED_CONTROLS
        or protocol.get("cross_lineage_evidence_splicing") != "PROHIBITED"
    ):
        raise Phase3Error("certificate-readiness governance changed")
    for relative, expected in protocol["bindings"].items():
        target = Path(relative) if Path(relative).is_absolute() else root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"certificate-readiness binding changed: {relative}")
    return protocol, sha256_file(path)


def _final_minus_parent(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    candidate = _rows(root / protocol["candidate_outputs"])
    parent = _rows(root / protocol["parent_outputs"])
    if set(candidate) != set(parent) or len(candidate) != 1400:
        raise Phase3Error("candidate/parent prompt pairing changed")
    rows = []
    counts = {capability: 0 for capability in CAPABILITIES}
    for probe_id, row in candidate.items():
        baseline = parent[probe_id]
        capability = str(row["capability"])
        if capability != str(baseline["capability"]) or capability not in counts:
            raise Phase3Error("candidate/parent capability join changed")
        counts[capability] += 1
        rows.append(
            {
                "capability": capability,
                "candidate_pass": bool(row["functional_pass_v1"]),
                "teacher_pass": bool(baseline["functional_pass"]),
            }
        )
    if any(value != 100 for value in counts.values()):
        raise Phase3Error("candidate/parent stratification changed")
    return paired_stratified_bootstrap(rows, replicates=10_000, seed=5151729)


def _evaluate_documents(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    docs = {name: _json(root / path) for name, path in protocol["documents"].items()}
    historical = docs["historical_controls"]
    quality = docs["final_quality"]
    verifier = docs["final_verifier"]
    replication = docs["replication"]
    runtime = docs["runtime"]
    accounting = docs["accounting"]
    accounting_verify = docs["accounting_verify"]

    endpoint = {
        "quality": all(bool(value) for value in quality["gates"].values()),
        "hostile_verifier": verifier["status"].startswith("PASS_")
        and int(verifier["evidence"]["adversarial_mutations_rejected"]) >= 5,
        "three_host_replication": replication["status"].startswith("PASS_")
        and int(replication["hosts"]) == 3
        and bool(replication["semantic_hashes_identical"]),
        "fully_cpu_runtime": runtime["gates"]["complete_corrected_gate_matrix"] == "17/17 PASS",
        "information_accounting": accounting_verify["status"].startswith("PASS_")
        and accounting["artifact"]["sha256"] == accounting_verify["artifact"]["sha256"],
    }
    historical_control = {
        "matched_A0_beats_A1_A2_A3_A4": bool(
            historical["gates"]["causal_A0_beats_each_control"]
        ),
        "historical_A0_absolute_quality": bool(
            historical["gates"]["per_capability_functional"]
            and historical["gates"]["critical_capabilities"]
            and historical["gates"]["zero_repetition_collapses"]
            and historical["gates"]["teacher_relative_noninferiority"]
        ),
        "same_checkpoint_as_final_candidate": historical["systems"]["A0"]["checkpoint_sha256"]
        == quality["candidate"]["checkpoint_sha256"],
    }
    return {
        "endpoint_gates": endpoint,
        "historical_control_gates": historical_control,
        "historical_control_evidence_can_certify_final": all(historical_control.values()),
    }


def execute(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = validate_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable audit output exists: {output}")
    evaluation = _evaluate_documents(root, protocol)
    parent = _final_minus_parent(root, protocol)

    # No final-lineage control evidence was registered at preregistration time.
    # Absence is a failed readiness gate, not permission to synthesize evidence.
    registered = tuple(protocol["registered_final_control_evidence"])
    control_gates = {
        "parent": parent["lower_95"] > 0.0,
        "A1_label_free": "A1_label_free" in registered,
        "A2_shuffled": "A2_shuffled" in registered,
        "A3_bridge_only": "A3_bridge_only" in registered,
        "A4_monolithic": "A4_monolithic" in registered,
    }
    endpoint_pass = all(evaluation["endpoint_gates"].values())
    causal_pass = all(control_gates.values()) and evaluation["historical_control_evidence_can_certify_final"]
    result = {
        "format": "abi-capability-compiler-phase3-certificate-readiness-result/1",
        "protocol_sha256": protocol_sha,
        "status": "PASS_PHASE3_CERTIFICATE_READY" if endpoint_pass and causal_pass else "FAIL_FINAL_LINEAGE_MATCHED_CAUSAL_CONTROLS_MISSING",
        "endpoint_gates": evaluation["endpoint_gates"],
        "historical_control_gates": evaluation["historical_control_gates"],
        "historical_control_evidence_can_certify_final": evaluation["historical_control_evidence_can_certify_final"],
        "final_candidate_minus_parent": parent,
        "final_lineage_control_gates": control_gates,
        "endpoint_qualified": endpoint_pass,
        "causal_controls_qualified": causal_pass,
        "phase3_certificate_ready": endpoint_pass and causal_pass,
        "missing_evidence": [name for name, passed in control_gates.items() if not passed],
        "final_test_accessed": False,
        "historical_evidence_changed": False,
        "evidence_sha256": "",
    }
    payload = dict(result)
    payload.pop("evidence_sha256")
    result["evidence_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    output.mkdir(parents=True)
    _write_immutable(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.root.resolve(), args.protocol.resolve(), args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
