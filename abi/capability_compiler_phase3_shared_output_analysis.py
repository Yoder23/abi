"""Recompute the Phase 3 shared-output decision from raw development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3_analysis import Phase3AnalysisError, stratified_bootstrap, wilson
from .capability_compiler_phase3_shared_output import EXPECTED_TRAINABLE_PARAMETERS, SYSTEMS, load_protocol


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3AnalysisError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compute_gates(systems: Mapping[str, Mapping[str, Any]], comparisons: Mapping[str, Mapping[str, float]], requirements: Mapping[str, Any]) -> dict[str, bool]:
    candidate = systems["C0"]
    caps = candidate["per_capability"]
    ordinary = all(
        value["wilson"]["point"] >= float(requirements["per_capability_functional_point_estimate_minimum"])
        and value["wilson"]["lower_95"] >= float(requirements["per_capability_functional_wilson_lower_minimum"])
        for value in caps.values()
    )
    critical = all(
        caps[name]["wilson"]["point"] >= float(requirements["prompt_grounding_instruction_adherence_and_abstention_point_minimum"])
        and caps[name]["wilson"]["lower_95"] >= float(requirements["prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum"])
        for name in ("prompt_grounding", "instruction_following", "abstention")
    )
    causal = all(
        comparisons[f"C0_minus_{control}"]["lower_95"]
        > float(requirements["C0_minus_each_C1_C2_C3_C4_paired_functional_bootstrap_lower_minimum"])
        for control in ("C1", "C2", "C3", "C4")
    )
    return {
        "per_capability_functional": ordinary,
        "critical_capabilities": critical,
        "zero_repetition_collapses": candidate["repetition_collapses"] <= int(requirements["repetition_collapse_count_maximum"]),
        "teacher_relative_noninferiority": comparisons["C0_minus_T0"]["lower_95"] >= float(requirements["teacher_relative_paired_difference_lower_bound_minimum"]),
        "causal_C0_beats_each_control": causal,
        "teacher_absent": True,
        "source_parameters_copied_zero": True,
        "registered_bridge_only": True,
        "final_test_not_accessed": True,
    }


def analyze(*, root: Path, protocol_path: Path, evidence_root: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        raise Phase3AnalysisError(f"analysis is immutable: {output_path}")
    protocol, protocol_sha = load_protocol(root, protocol_path)
    systems: dict[str, Any] = {}
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    sequence_hashes = set()
    expected_controls = {
        "C0": (True, False, True, False),
        "C1": (False, False, True, False),
        "C2": (True, True, True, False),
        "C3": (True, False, False, False),
        "C4": (False, False, True, True),
    }
    for system in SYSTEMS:
        candidate = evidence_root / "development_v11" / f"{system}-seed104729"
        evaluation = evidence_root / "evaluation_v11" / f"{system}-seed104729"
        metadata_path = candidate / "metadata.json"
        receipt_path = evaluation / "receipt.json"
        outputs_path = evaluation / "development_outputs.jsonl"
        metadata = _json(metadata_path)
        receipt = _json(receipt_path)
        rows = _jsonl(outputs_path)
        control = metadata.get("control", {})
        observed_control = (
            control.get("uses_destination_labels"),
            control.get("targets_deranged"),
            control.get("teacher_payload_present"),
            control.get("monolithic_route"),
        )
        isolation = metadata.get("isolation", {})
        source = metadata.get("source", {})
        training = metadata.get("training", {})
        if (
            metadata.get("system") != system
            or receipt.get("system") != system
            or metadata.get("seed") != 104729
            or receipt.get("seed") != 104729
            or metadata.get("protocol_sha256") != protocol_sha
            or receipt.get("protocol_sha256") != protocol_sha
            or metadata.get("final_test_accessed") is not False
            or receipt.get("final_test_accessed") is not False
            or sha256_file(outputs_path) != receipt.get("outputs_sha256")
            or len(rows) != 1400
            or len({row["probe_id"] for row in rows}) != 1400
            or sum(bool(row["functional_pass"]) for row in rows) != receipt.get("functional_passes")
            or sum(bool(row["repetition_collapse"]) for row in rows) != receipt.get("repetition_collapses")
            or training.get("trainable_parameters") != EXPECTED_TRAINABLE_PARAMETERS
            or training.get("wrong_repeat_loss_weight") != protocol["training"]["wrong_repeat_loss_weight"]
            or isolation.get("all_changes_confined_to_registered_bridge") is not True
            or isolation.get("frozen_state_sha256_before") != isolation.get("frozen_state_sha256_after")
            or source.get("teacher_present_during_training") is not False
            or source.get("teacher_present_at_inference") is not False
            or source.get("source_parameters_copied") != 0
            or source.get("source_blocks_retained") != 0
            or observed_control != expected_controls[system]
        ):
            raise Phase3AnalysisError(f"{system} evidence failed identity or aggregate verification")
        by_id = {str(row["probe_id"]): row for row in rows}
        raw[system] = by_id
        per_capability = {}
        for capability in CAPABILITIES:
            values = [row for row in rows if row["capability"] == capability]
            if len(values) != 100:
                raise Phase3AnalysisError(f"{system} capability depth changed")
            passes = sum(bool(row["functional_pass"]) for row in values)
            per_capability[capability] = {
                "passes": passes,
                "observations": 100,
                "collapses": sum(bool(row["repetition_collapse"]) for row in values),
                "wilson": wilson(passes, 100),
            }
        sequence = training.get("successful_record_sequence_sha256")
        sequence_hashes.add(sequence)
        systems[system] = {
            "checkpoint_sha256": metadata["checkpoint"]["sha256"],
            "metadata_sha256": sha256_file(metadata_path),
            "receipt_sha256": sha256_file(receipt_path),
            "outputs_sha256": receipt["outputs_sha256"],
            "functional_passes": receipt["functional_passes"],
            "functional": wilson(receipt["functional_passes"], 1400),
            "repetition_collapses": receipt["repetition_collapses"],
            "per_capability": per_capability,
            "training_wall_seconds": training["wall_seconds"],
            "generation_wall_seconds": receipt["wall_seconds"],
            "teacher_response_tokens_seen": training["teacher_response_tokens_seen"],
            "wrong_repeat_penalty_events": training["wrong_repeat_penalty_events"],
            "trainable_parameters": training["trainable_parameters"],
            "peak_process_rss_bytes": training["peak_process_rss_bytes"],
            "peak_cuda_allocated_bytes": training["peak_cuda_allocated_bytes"],
            "skipped_amp_steps": training["skipped_amp_steps"],
            "successful_record_sequence_sha256": sequence,
        }
    if len(sequence_hashes) != 1 or None in sequence_hashes:
        raise Phase3AnalysisError("successful paired record sequences differ")
    teacher_rows = _jsonl(root / protocol["development"]["teacher_reference_path"])
    teacher = {str(row["probe_id"]): row for row in teacher_rows}
    if len(teacher) != 1400:
        raise Phase3AnalysisError("teacher reference depth changed")
    comparisons = {
        f"C0_minus_{control}": stratified_bootstrap(raw["C0"], raw[control], replicates=10000, seed=3141)
        for control in ("C1", "C2", "C3", "C4")
    }
    comparisons["C0_minus_T0"] = stratified_bootstrap(raw["C0"], teacher, replicates=10000, seed=3141)
    gates = compute_gates(systems, comparisons, protocol["automated_pass_requirements"])
    initial_pass = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-shared-output-decision/1",
        "status": "PASS_INITIAL_SEED_REMAINING_SEEDS_AUTHORIZED" if initial_pass else "FAIL_INITIAL_SEED_SHARED_OUTPUT_SUCCESSOR",
        "protocol": {"path": protocol_path.name, "sha256": protocol_sha},
        "phase2_status": "MACHINE_COMPLETE_HUMAN_RATINGS_DEFERRED_NOT_PASSED",
        "phase3_certified": False,
        "phase4_status": "LOCKED",
        "systems": systems,
        "paired_bootstrap": comparisons,
        "gates": gates,
        "decision": {
            "branch_promoted": False,
            "remaining_two_seeds_authorized": initial_pass,
            "reason": "C0 passed every initial automated gate; run only preregistered paired seeds." if initial_pass else "C0 failed one or more locked initial-seed gates; close this exact branch.",
        },
        "negative_evidence_preserved": True,
        "final_test_accessed": False,
        "claim_boundary": "Initial success authorizes reproduction only; failure closes this branch. Neither certifies Phase 3 while Phase 2 human ratings are deferred.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_SHARED_OUTPUT_PROTOCOL_V11.json")
    parser.add_argument("--evidence-root", default="results/abi_capability_compiler_phase3_shared_output")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_shared_output/conditional_decision_v1.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = analyze(root=root, protocol_path=(root / args.protocol).resolve(), evidence_root=(root / args.evidence_root).resolve(), output_path=(root / args.output).resolve())
    print(json.dumps({"status": result["status"], "phase3_certified": False, "phase4_status": "LOCKED", "evidence_sha256": result["evidence_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
