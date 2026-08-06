"""Recompute the V17 self-prefix successor decision from raw prompt evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3_analysis import Phase3AnalysisError, stratified_bootstrap, wilson
from .capability_compiler_phase3_self_prefix import SYSTEMS, load_protocol


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise Phase3AnalysisError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def analyze(root: Path, protocol_path: Path, evidence_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists(): raise Phase3AnalysisError("self-prefix decision is immutable")
    protocol, protocol_sha = load_protocol(root, protocol_path)
    systems = {}; raw = {}; sample_hashes = set(); event_counts = set(); token_counts = set(); policy_hashes = set()
    evidence_version = "v18" if "recent-repeat" in str(protocol.get("protocol_id", "")) else "v17"
    for system in SYSTEMS:
        candidate = evidence_root / f"development_{evidence_version}" / f"{system}-seed190081"
        evaluation = evidence_root / f"evaluation_{evidence_version}" / f"{system}-seed190081"
        metadata_path = candidate / "metadata.json"; receipt_path = evaluation / "receipt.json"; outputs_path = evaluation / "development_outputs.jsonl"
        metadata = _json(metadata_path); receipt = _json(receipt_path); rows = _jsonl(outputs_path)
        training = metadata.get("training", {}); isolation = metadata.get("isolation", {})
        expected_weight = 0.25 if system == "S0" else 0.0
        if (
            metadata.get("system") != system or receipt.get("system") != system
            or metadata.get("protocol_sha256") != protocol_sha or receipt.get("protocol_sha256") != protocol_sha
            or metadata.get("checkpoint", {}).get("sha256") != receipt.get("checkpoint_sha256")
            or sha256_file(candidate / "model.safetensors") != metadata.get("checkpoint", {}).get("sha256")
            or sha256_file(outputs_path) != receipt.get("outputs_sha256") or len(rows) != 1400
            or len({row["probe_id"] for row in rows}) != 1400
            or sum(bool(row["functional_pass"]) for row in rows) != receipt.get("functional_passes")
            or sum(bool(row["repetition_collapse"]) for row in rows) != receipt.get("repetition_collapses")
            or training.get("steps") != 1000 or training.get("self_prefix_weight") != expected_weight
            or training.get("compute_matched_corrupted_forward") is not True
            or isolation.get("all_changes_confined_to_registered_bridge") is not True
            or isolation.get("frozen_state_sha256_before") != isolation.get("frozen_state_sha256_after")
            or isolation.get("corruption_policy_state_sha256_before") != isolation.get("corruption_policy_state_sha256_after")
            or metadata.get("teacher_present_at_training") is not False or metadata.get("teacher_present_at_inference") is not False
            or metadata.get("final_test_accessed") is not False or receipt.get("final_test_accessed") is not False
        ): raise Phase3AnalysisError(f"{system} evidence failed identity or isolation checks")
        sample_hashes.add(training["successful_record_sequence_sha256"]); event_counts.add(training["recovery_events"]); token_counts.add(training["recovery_tokens"]); policy_hashes.add(isolation["corruption_policy_state_sha256_before"])
        raw[system] = {str(row["probe_id"]): row for row in rows}
        per_capability = {}
        for capability in CAPABILITIES:
            values = [row for row in rows if row["capability"] == capability]
            if len(values) != 100: raise Phase3AnalysisError("capability depth changed")
            passes = sum(bool(row["functional_pass"]) for row in values)
            per_capability[capability] = {"passes": passes, "collapses": sum(bool(row["repetition_collapse"]) for row in values), "wilson": wilson(passes, 100)}
        systems[system] = {
            "checkpoint_sha256": metadata["checkpoint"]["sha256"], "metadata_sha256": sha256_file(metadata_path), "receipt_sha256": sha256_file(receipt_path), "outputs_sha256": receipt["outputs_sha256"],
            "functional_passes": receipt["functional_passes"], "functional": wilson(receipt["functional_passes"], 1400), "repetition_collapses": receipt["repetition_collapses"], "per_capability": per_capability,
            "training_wall_seconds": training["wall_seconds"], "generation_wall_seconds": receipt["wall_seconds"], "recovery_events": training["recovery_events"], "recovery_tokens": training["recovery_tokens"],
        }
    if any(len(values) != 1 for values in (sample_hashes, event_counts, token_counts, policy_hashes)): raise Phase3AnalysisError("paired training inputs or frozen policy differ")
    c0_path = root / "results/abi_capability_compiler_phase3_shared_output/evaluation_v11/C0-seed104729/development_outputs.jsonl"
    teacher_path = root / "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl"
    c0 = {str(row["probe_id"]): row for row in _jsonl(c0_path)}; teacher = {str(row["probe_id"]): row for row in _jsonl(teacher_path)}
    comparisons = {
        "S0_minus_S1": stratified_bootstrap(raw["S0"], raw["S1"], replicates=10000, seed=2718),
        "S0_minus_V11_C0": stratified_bootstrap(raw["S0"], c0, replicates=10000, seed=2718),
        "S0_minus_T0": stratified_bootstrap(raw["S0"], teacher, replicates=10000, seed=2718),
    }
    req = protocol["decision_gates"]; caps = systems["S0"]["per_capability"]
    gates = {
        "S0_beats_compute_matched_S1": comparisons["S0_minus_S1"]["lower_95"] > float(req["S0_minus_S1_paired_functional_bootstrap_lower_minimum"]),
        "S0_beats_V11_C0": comparisons["S0_minus_V11_C0"]["lower_95"] > float(req["S0_minus_V11_C0_paired_functional_bootstrap_lower_minimum"]),
        "per_capability_functional": all(v["wilson"]["point"] >= float(req["per_capability_functional_point_estimate_minimum"]) for v in caps.values()),
        "critical_capabilities": all(caps[name]["wilson"]["point"] >= float(req["prompt_grounding_instruction_adherence_and_abstention_point_minimum"]) for name in ("prompt_grounding", "instruction_following", "abstention")),
        "teacher_relative_noninferiority": comparisons["S0_minus_T0"]["lower_95"] >= float(req["teacher_relative_paired_difference_lower_bound_minimum"]),
        "zero_repetition_collapses": systems["S0"]["repetition_collapses"] <= int(req["repetition_collapse_count_maximum"]),
        "paired_training_sequence": len(sample_hashes) == 1, "frozen_host_and_policy": True, "teacher_absent": True, "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-self-prefix-decision/1", "status": "PASS_INITIAL_SEED_REMAINING_SEEDS_AUTHORIZED" if passed else "FAIL_INITIAL_SEED_SELF_PREFIX_SUCCESSOR",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha}, "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED", "phase3_certified": False, "phase4_status": "LOCKED",
        "systems": systems, "paired_bootstrap": comparisons, "gates": gates,
        "decision": {"branch_promoted": False, "remaining_two_seeds_authorized": passed, "reason": "All initial gates passed; run only registered paired seeds." if passed else "S0 failed one or more locked gates; close V17 without tuning."},
        "paired_training": {"successful_record_sequence_sha256": next(iter(sample_hashes)), "recovery_events": next(iter(event_counts)), "recovery_tokens": next(iter(token_counts)), "frozen_policy_state_sha256": next(iter(policy_hashes))},
        "negative_evidence_preserved": True, "final_test_accessed": False, "abi_superiority_claim_allowed": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SELF_PREFIX_PROTOCOL_V17.json"); parser.add_argument("--evidence-root", default="results/abi_capability_compiler_phase3_self_prefix"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_self_prefix/conditional_decision_v1.json"); args = parser.parse_args(argv)
    root = Path.cwd().resolve(); result = analyze(root, (root / args.protocol).resolve(), (root / args.evidence_root).resolve(), (root / args.output).resolve()); print(json.dumps({"status": result["status"], "gates": result["gates"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
