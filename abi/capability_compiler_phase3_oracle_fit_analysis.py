"""Recompute the permanently non-promotional V20 oracle-fit decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3_oracle_fit import load_protocol


class OracleAnalysisError(ValueError): pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise OracleAnalysisError(f"expected object: {path}")
    return value


def analyze(root: Path, protocol_path: Path, candidate_dir: Path, evaluation_dir: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists(): raise OracleAnalysisError("oracle decision is immutable")
    protocol, protocol_sha = load_protocol(root, protocol_path); metadata_path = candidate_dir / "metadata.json"; receipt_path = evaluation_dir / "receipt.json"; outputs_path = evaluation_dir / "development_outputs.jsonl"; metadata = _json(metadata_path); receipt = _json(receipt_path)
    rows = [json.loads(line) for line in outputs_path.read_text(encoding="utf-8").splitlines() if line]
    isolation = metadata.get("isolation", {})
    if (
        metadata.get("protocol_sha256") != protocol_sha or receipt.get("protocol_sha256") != protocol_sha
        or metadata.get("development_contaminated") is not True or receipt.get("development_contaminated") is not True
        or metadata.get("promotion_eligible") is not False or receipt.get("promotion_eligible") is not False
        or metadata.get("final_test_accessed") is not False or receipt.get("final_test_accessed") is not False
        or sha256_file(candidate_dir / "model.safetensors") != metadata.get("checkpoint", {}).get("sha256")
        or metadata.get("checkpoint", {}).get("sha256") != receipt.get("checkpoint_sha256")
        or sha256_file(outputs_path) != receipt.get("outputs_sha256") or len(rows) != 1400 or len({r["probe_id"] for r in rows}) != 1400
        or sum(bool(r["functional_pass"]) for r in rows) != receipt.get("functional_passes")
        or sum(bool(r["repetition_collapse"]) for r in rows) != receipt.get("repetition_collapses")
        or isolation.get("all_changes_confined_to_registered_bridge") is not True
        or isolation.get("frozen_state_sha256_before") != isolation.get("frozen_state_sha256_after")
    ): raise OracleAnalysisError("oracle identity, contamination, aggregate, or isolation check failed")
    per_capability = {}
    for capability in CAPABILITIES:
        values = [r for r in rows if r["capability"] == capability]
        if len(values) != 100: raise OracleAnalysisError("oracle capability depth changed")
        per_capability[capability] = {"passes": sum(bool(r["functional_pass"]) for r in values), "collapses": sum(bool(r["repetition_collapse"]) for r in values)}
    gate = protocol["capacity_sufficient_gate"]
    gates = {
        "aggregate_functional": receipt["functional_passes"] / 1400 >= float(gate["aggregate_functional_rate_minimum"]),
        "per_capability_functional": all(v["passes"] / 100 >= float(gate["per_capability_functional_rate_minimum"]) for v in per_capability.values()),
        "zero_repetition_collapses": receipt["repetition_collapses"] <= int(gate["repetition_collapse_count_maximum"]),
        "frozen_host_unchanged": True, "autonomous_generation": True, "development_contaminated": True, "promotion_prohibited": True, "final_test_not_accessed": True,
    }
    sufficient = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-oracle-fit-decision/1", "status": "CAPACITY_SUFFICIENT_DIAGNOSTIC_ONLY" if sufficient else "HOST_BRIDGE_EXPRESSIVITY_OR_OPTIMIZATION_LIMITATION",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha}, "checkpoint_sha256": metadata["checkpoint"]["sha256"],
        "functional_passes": receipt["functional_passes"], "observations": 1400, "repetition_collapses": receipt["repetition_collapses"], "per_capability": per_capability, "gates": gates,
        "decision": {"capacity_sufficient": sufficient, "next_action": "return to ABI acquisition generalization" if sufficient else "run one read-only oracle teacher-forced fit diagnostic, then hand the measured limitation to a separately governed LayerCake host/bridge investigation"},
        "metadata_sha256": sha256_file(metadata_path), "receipt_sha256": sha256_file(receipt_path), "outputs_sha256": sha256_file(outputs_path),
        "development_contaminated": True, "promotion_eligible": False, "phase3_certified": False, "phase4_status": "LOCKED", "final_test_accessed": False, "abi_superiority_claim_allowed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ORACLE_FIT_PROTOCOL_V20.json"); parser.add_argument("--candidate-dir", default="results/abi_capability_compiler_phase3_oracle_fit/development_v20/O0-seed230003"); parser.add_argument("--evaluation-dir", default="results/abi_capability_compiler_phase3_oracle_fit/evaluation_v20/O0-seed230003"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_oracle_fit/decision_v1.json"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = analyze(root, (root / args.protocol).resolve(), (root / args.candidate_dir).resolve(), (root / args.evaluation_dir).resolve(), (root / args.output).resolve()); print(json.dumps({"status": result["status"], "gates": result["gates"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
