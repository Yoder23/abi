"""Recompute the Phase 3 sequence-successor decision from raw development evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, sha256_file
from .capability_compiler_phase3_analysis import (
    Phase3AnalysisError,
    stratified_bootstrap,
    wilson,
)
from .capability_compiler_phase3_sequence_bridge import (
    EXPECTED_TRAINABLE_PARAMETERS,
    SYSTEMS,
    load_protocol,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3AnalysisError(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def compute_gates(
    systems: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, float]],
    requirements: Mapping[str, Any],
) -> dict[str, bool]:
    a0 = systems["B0"]
    caps = a0["per_capability"]
    ordinary = all(
        value["wilson"]["point"]
        >= float(requirements["per_capability_functional_point_estimate_minimum"])
        and value["wilson"]["lower_95"]
        >= float(requirements["per_capability_functional_wilson_lower_minimum"])
        for value in caps.values()
    )
    critical = all(
        caps[name]["wilson"]["point"]
        >= float(
            requirements[
                "prompt_grounding_instruction_adherence_and_abstention_point_minimum"
            ]
        )
        and caps[name]["wilson"]["lower_95"]
        >= float(
            requirements[
                "prompt_grounding_instruction_adherence_and_abstention_wilson_lower_minimum"
            ]
        )
        for name in ("prompt_grounding", "instruction_following", "abstention")
    )
    causal = all(
        comparisons[f"B0_minus_{control}"]["lower_95"]
        > float(
            requirements[
                "B0_minus_each_B1_B2_B3_B4_paired_functional_bootstrap_lower_minimum"
            ]
        )
        for control in ("B1", "B2", "B3", "B4")
    )
    return {
        "per_capability_functional": ordinary,
        "critical_capabilities": critical,
        "zero_repetition_collapses": a0["repetition_collapses"]
        <= int(requirements["repetition_collapse_count_maximum"]),
        "teacher_relative_noninferiority": comparisons["B0_minus_T0"]["lower_95"]
        >= float(requirements["teacher_relative_paired_difference_lower_bound_minimum"]),
        "causal_B0_beats_each_control": causal,
        "teacher_absent": True,
        "source_parameters_copied_zero": True,
        "registered_bridge_only": True,
        "final_test_not_accessed": True,
    }


def analyze(
    *, root: Path, protocol_path: Path, evidence_root: Path, output_path: Path
) -> dict[str, Any]:
    if output_path.exists():
        raise Phase3AnalysisError(f"analysis is immutable: {output_path}")
    protocol, protocol_sha = load_protocol(root, protocol_path)
    systems: dict[str, Any] = {}
    raw: dict[str, dict[str, dict[str, Any]]] = {}
    sequence_hashes = set()
    for system in SYSTEMS:
        candidate = evidence_root / "development_v6" / f"{system}-seed104729"
        evaluation = evidence_root / "evaluation_v6" / f"{system}-seed104729"
        metadata_path = candidate / "metadata.json"
        receipt_path = evaluation / "receipt.json"
        outputs_path = evaluation / "development_outputs.jsonl"
        metadata = _json(metadata_path)
        receipt = _json(receipt_path)
        rows = _jsonl(outputs_path)
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
            or sum(bool(row["functional_pass"]) for row in rows)
            != receipt.get("functional_passes")
            or sum(bool(row["repetition_collapse"]) for row in rows)
            != receipt.get("repetition_collapses")
            or metadata["training"].get("trainable_parameters")
            != EXPECTED_TRAINABLE_PARAMETERS
            or metadata["isolation"].get("all_changes_confined_to_registered_bridge")
            is not True
            or metadata["source"].get("teacher_present_at_inference") is not False
            or metadata["source"].get("source_parameters_copied") != 0
        ):
            raise Phase3AnalysisError(f"{system} evidence failed identity or aggregate verification")
        by_id = {str(row["probe_id"]): row for row in rows}
        if len(by_id) != 1400:
            raise Phase3AnalysisError(f"{system} prompt identities are not unique")
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
        sequence = metadata["training"].get("successful_record_sequence_sha256")
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
            "training_wall_seconds": metadata["training"]["wall_seconds"],
            "generation_wall_seconds": receipt["wall_seconds"],
            "teacher_response_tokens_seen": metadata["training"]["teacher_response_tokens_seen"],
            "trainable_parameters": metadata["training"]["trainable_parameters"],
            "peak_process_rss_bytes": metadata["training"]["peak_process_rss_bytes"],
            "peak_cuda_allocated_bytes": metadata["training"]["peak_cuda_allocated_bytes"],
            "skipped_amp_steps": metadata["training"]["skipped_amp_steps"],
            "successful_record_sequence_sha256": sequence,
        }
    if len(sequence_hashes) != 1 or None in sequence_hashes:
        raise Phase3AnalysisError("successful paired record sequences differ")
    teacher_rows = _jsonl(root / protocol["development"]["teacher_reference_path"])
    teacher = {str(row["probe_id"]): row for row in teacher_rows}
    if len(teacher) != 1400:
        raise Phase3AnalysisError("teacher reference depth changed")
    comparisons = {
        f"B0_minus_{control}": stratified_bootstrap(
            raw["B0"], raw[control], replicates=10_000, seed=1729
        )
        for control in ("B1", "B2", "B3", "B4")
    }
    comparisons["B0_minus_T0"] = stratified_bootstrap(
        raw["B0"], teacher, replicates=10_000, seed=1729
    )
    gates = compute_gates(systems, comparisons, protocol["automated_pass_requirements"])
    initial_pass = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-sequence-successor-decision/1",
        "status": (
            "PASS_INITIAL_SEED_REMAINING_SEEDS_AUTHORIZED"
            if initial_pass
            else "FAIL_INITIAL_SEED_SEQUENCE_SUCCESSOR"
        ),
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
            "reason": (
                "B0 passed every initial-seed automated gate; run only the two preregistered paired seeds."
                if initial_pass
                else "B0 failed one or more locked initial-seed gates; stop this exact branch and preserve all evidence."
            ),
        },
        "negative_evidence_preserved": True,
        "final_test_accessed": False,
        "claim_boundary": (
            "An initial-seed pass authorizes paired reproduction only; it does not certify Phase 3 while Phase 2 human ratings are deferred."
            if initial_pass
            else "This is a failed development architecture, not Phase 3 completion or ABI superiority."
        ),
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_SEQUENCE_SUCCESSOR_PROTOCOL_V6.json",
    )
    parser.add_argument(
        "--evidence-root",
        default="results/abi_capability_compiler_phase3_sequence",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_sequence/conditional_decision_v1.json",
    )
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = analyze(
        root=root,
        protocol_path=(root / args.protocol).resolve(),
        evidence_root=(root / args.evidence_root).resolve(),
        output_path=(root / args.output).resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase3_certified": result["phase3_certified"],
                "phase4_status": result["phase4_status"],
                "evidence_sha256": result["evidence_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
