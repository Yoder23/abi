"""Recompute and verify the non-promotional V22 expanded-bridge oracle decision."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error
from . import capability_compiler_phase3_oracle_fit as oracle
from .capability_compiler_phase3_expanded_oracle import (
    EXPANDED_TRAINABLE_PARAMETERS,
    _install_delegate,
)


class ExpandedOracleAnalysisError(Phase3Error):
    """Raised when V22 evidence does not reproduce exactly."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExpandedOracleAnalysisError(f"expected object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise ExpandedOracleAnalysisError("output row is not an object")
    return rows


def compute(
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    evaluation_dir: Path,
) -> dict[str, Any]:
    _install_delegate()
    protocol, protocol_sha = oracle.load_protocol(root, protocol_path)
    metadata_path = candidate_dir / "metadata.json"
    receipt_path = evaluation_dir / "receipt.json"
    outputs_path = evaluation_dir / "development_outputs.jsonl"
    checkpoint_path = candidate_dir / "model.safetensors"
    metadata = _json(metadata_path)
    receipt = _json(receipt_path)
    rows = _load_rows(outputs_path)
    isolation = metadata.get("isolation", {})
    checkpoint_sha = sha256_file(checkpoint_path)
    output_sha = sha256_file(outputs_path)

    checks = {
        "protocol_metadata": metadata.get("protocol_sha256") == protocol_sha,
        "protocol_receipt": receipt.get("protocol_sha256") == protocol_sha,
        "development_contaminated": metadata.get("development_contaminated") is True
        and receipt.get("development_contaminated") is True,
        "promotion_prohibited": metadata.get("promotion_eligible") is False
        and receipt.get("promotion_eligible") is False,
        "final_not_accessed": metadata.get("final_test_accessed") is False
        and receipt.get("final_test_accessed") is False,
        "checkpoint_bound": checkpoint_sha == metadata.get("checkpoint", {}).get("sha256")
        == receipt.get("checkpoint_sha256"),
        "outputs_bound": output_sha == receipt.get("outputs_sha256"),
        "depth": len(rows) == 1400 and len({row.get("probe_id") for row in rows}) == 1400,
        "aggregate": sum(bool(row.get("functional_pass")) for row in rows)
        == receipt.get("functional_passes"),
        "collapse_aggregate": sum(bool(row.get("repetition_collapse")) for row in rows)
        == receipt.get("repetition_collapses"),
        "frozen_host": isolation.get("all_changes_confined_to_registered_bridge") is True
        and isolation.get("frozen_state_sha256_before")
        == isolation.get("frozen_state_sha256_after"),
        "trainable_count": protocol["training"]["trainable_parameters"]
        == EXPANDED_TRAINABLE_PARAMETERS,
        "speed_not_inherited": protocol["architecture"].get("speed_claim_inherited") is False,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ExpandedOracleAnalysisError(f"V22 identity/evidence checks failed: {failed}")

    per_capability: dict[str, dict[str, int]] = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row.get("capability") == capability]
        if len(values) != 100:
            raise ExpandedOracleAnalysisError(f"V22 capability depth changed: {capability}")
        per_capability[capability] = {
            "passes": sum(bool(row.get("functional_pass")) for row in values),
            "collapses": sum(bool(row.get("repetition_collapse")) for row in values),
        }
        reported = receipt.get("per_capability", {}).get(capability, {})
        if reported.get("passes") != per_capability[capability]["passes"] or reported.get(
            "collapses"
        ) != per_capability[capability]["collapses"]:
            raise ExpandedOracleAnalysisError(f"V22 per-capability receipt mismatch: {capability}")

    gate = protocol["capacity_sufficient_gate"]
    gates = {
        "aggregate_functional": receipt["functional_passes"] / 1400
        >= float(gate["aggregate_functional_rate_minimum"]),
        "per_capability_functional": all(
            value["passes"] / 100 >= float(gate["per_capability_functional_rate_minimum"])
            for value in per_capability.values()
        ),
        "zero_repetition_collapses": receipt["repetition_collapses"]
        <= int(gate["repetition_collapse_count_maximum"]),
        "frozen_host_unchanged": True,
        "autonomous_generation": True,
        "development_contaminated": True,
        "promotion_prohibited": True,
        "final_test_not_accessed": True,
        "speed_claim_not_inherited": True,
    }
    sufficient = all(gates.values())
    result: dict[str, Any] = {
        "format": "abi-capability-compiler-phase3-expanded-oracle-decision/1",
        "status": "CAPACITY_SUFFICIENT_DIAGNOSTIC_ONLY"
        if sufficient
        else "EXPANDED_INTEGRATION_BRIDGE_INSUFFICIENT",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "checkpoint_sha256": checkpoint_sha,
        "functional_passes": receipt["functional_passes"],
        "observations": 1400,
        "functional_rate": receipt["functional_passes"] / 1400,
        "repetition_collapses": receipt["repetition_collapses"],
        "per_capability": per_capability,
        "gates": gates,
        "comparison_to_v20_oracle": {
            "v20_functional_passes": 1229,
            "functional_pass_delta": receipt["functional_passes"] - 1229,
            "v20_repetition_collapses": 89,
            "repetition_collapse_delta": receipt["repetition_collapses"] - 89,
        },
        "decision": {
            "capacity_sufficient": sufficient,
            "abi_acquisition_experimentation_authorized": sufficient,
            "next_action": "fresh-data production acquisition candidate with independent speed recertification"
            if sufficient
            else "stop ABI acquisition experiments and open a separately governed LayerCake integration investigation",
        },
        "attribution": {
            "abi_extraction_failure_proven": False,
            "abi_labeling_failure_proven": False,
            "sealed_layercake_regression": False,
            "layercake_host_representational_ceiling_proven": False,
            "current_and_expanded_abi_integration_bridges_sufficient": sufficient,
        },
        "identity_checks": checks,
        "metadata_sha256": sha256_file(metadata_path),
        "receipt_sha256": sha256_file(receipt_path),
        "outputs_sha256": output_sha,
        "development_contaminated": True,
        "promotion_eligible": False,
        "phase3_certified": False,
        "phase4_status": "LOCKED",
        "final_test_accessed": False,
        "abi_superiority_claim_allowed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def analyze(
    root: Path,
    protocol_path: Path,
    candidate_dir: Path,
    evaluation_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise ExpandedOracleAnalysisError("V22 decision is immutable")
    result = compute(root, protocol_path, candidate_dir, evaluation_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def verify_decision(expected: Mapping[str, Any], recomputed: Mapping[str, Any]) -> None:
    if dict(expected) != dict(recomputed):
        raise ExpandedOracleAnalysisError("sealed V22 decision does not match raw evidence")


def adversarial_checks(recomputed: Mapping[str, Any]) -> dict[str, Any]:
    mutations = (
        ("functional_passes", lambda value: value.__setitem__("functional_passes", 1400)),
        ("collapses", lambda value: value.__setitem__("repetition_collapses", 0)),
        ("capacity", lambda value: value["decision"].__setitem__("capacity_sufficient", True)),
        ("phase3", lambda value: value.__setitem__("phase3_certified", True)),
        ("superiority", lambda value: value.__setitem__("abi_superiority_claim_allowed", True)),
        ("host_regression", lambda value: value["attribution"].__setitem__("sealed_layercake_regression", True)),
    )
    rejected = []
    for name, mutate in mutations:
        candidate = copy.deepcopy(dict(recomputed))
        mutate(candidate)
        try:
            verify_decision(candidate, recomputed)
        except ExpandedOracleAnalysisError:
            rejected.append(name)
    if len(rejected) != len(mutations):
        raise ExpandedOracleAnalysisError("V22 adversarial verifier accepted a mutation")
    return {"status": "PASS", "mutations_rejected": rejected, "count": len(rejected)}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "verify", "adversarial"))
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_EXPANDED_ORACLE_PROTOCOL_V22.json")
    parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_expanded_oracle/development_v22/O1-seed230003")
    parser.add_argument("--evaluation-dir", default="results/abi_capability_compiler_phase3_expanded_oracle/evaluation_v22/O1-seed230003")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_expanded_oracle/decision_v1.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    recomputed = compute(root, (root / args.protocol).resolve(), (root / args.candidate_dir).resolve(), (root / args.evaluation_dir).resolve())
    if args.mode == "analyze":
        result = analyze(root, (root / args.protocol).resolve(), (root / args.candidate_dir).resolve(), (root / args.evaluation_dir).resolve(), (root / args.output).resolve())
    elif args.mode == "verify":
        expected = _json((root / args.output).resolve())
        verify_decision(expected, recomputed)
        result = {"status": "PASS", "evidence_sha256": recomputed["evidence_sha256"]}
    else:
        result = adversarial_checks(recomputed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
