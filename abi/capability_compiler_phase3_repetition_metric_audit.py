"""V446/V447 read-only construct-validity audit for repetition metric v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .capability_compiler_phase2_common import repetition_collapse
from .capability_compiler_repetition_v2 import repetition_collapse_v2


SYSTEMS = {
    "teacher_T0": "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl",
    "candidate_V443": "results/abi_capability_compiler_phase3_qualified_transition_control/evaluation_v443/development_outputs.jsonl",
    "failed_A3": "results/abi_capability_compiler_phase3/evaluation_v4/A3-seed104729/development_outputs.jsonl",
    "failed_C3": "results/abi_capability_compiler_phase3_shared_output/evaluation_v11/C3-seed104729/development_outputs.jsonl",
    "failed_D2": "results/abi_capability_compiler_phase2/full_development/D2/lr3e-5-exp4-seed104729/development_outputs.jsonl",
}

NEGATIVE_REFERENCES = {
    "candidate_V443": [
        "phase1-validation-conversation-0004-v2",
        "phase1-validation-email_drafting-0000-v2",
    ],
    "teacher_T0": [
        "phase1-validation-instruction_following-0022-v2",
        "phase1-validation-instruction_following-0026-v2",
        "phase1-validation-instruction_following-0050-v2",
    ],
}

ACTUAL_POSITIVE_REFERENCE = (
    "candidate_V443",
    "phase1-validation-cake_output_realization-0063-v2",
)

SYNTHETIC_POSITIVES = [
    "echo " * 10,
    "alpha beta gamma " * 4,
    "8: " * 5,
    "." * 40,
    "the answer is blue. " * 4,
    "A B C D E F\n" * 4,
]

SYNTHETIC_NEGATIVES = [
    "The two visitors are waiting in the quiet room.",
    "I understand that this is frustrating. We can review the options together and choose a practical next step.",
    "Subject: Updated schedule\n\nHello team,\n\nThe review is Thursday at 2 p.m. Please send comments by Wednesday.\n\nBest,\nMira",
    "First, verify the input. Second, compute the result. Finally, report any assumptions clearly.",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def run(root: Path, protocol: Path, output: Path) -> dict[str, Any]:
    system_rows: dict[str, list[dict[str, Any]]] = {}
    systems: dict[str, Any] = {}
    for name, relative in SYSTEMS.items():
        path = root / relative
        rows = _rows(path)
        system_rows[name] = rows
        systems[name] = {
            "path": relative,
            "sha256": _sha256(path),
            "observations": len(rows),
            "historical_v1_collapses": sum(bool(row["repetition_collapse"]) for row in rows),
            "recomputed_v1_collapses": sum(repetition_collapse(str(row["output"])) for row in rows),
            "v2_collapses": sum(repetition_collapse_v2(str(row["output"])) for row in rows),
        }

    referenced_negatives: list[dict[str, Any]] = []
    for system, probe_ids in NEGATIVE_REFERENCES.items():
        indexed = {str(row["probe_id"]): row for row in system_rows[system]}
        for probe_id in probe_ids:
            row = indexed[probe_id]
            referenced_negatives.append(
                {
                    "system": system,
                    "probe_id": probe_id,
                    "output_sha256": hashlib.sha256(str(row["output"]).encode("utf-8")).hexdigest(),
                    "v1": repetition_collapse(str(row["output"])),
                    "v2": repetition_collapse_v2(str(row["output"])),
                }
            )

    positive_system, positive_probe_id = ACTUAL_POSITIVE_REFERENCE
    actual_positive = next(
        row
        for row in system_rows[positive_system]
        if str(row["probe_id"]) == positive_probe_id
    )
    controls = {
        "synthetic_positive_passes": sum(repetition_collapse_v2(value) for value in SYNTHETIC_POSITIVES),
        "synthetic_positive_observations": len(SYNTHETIC_POSITIVES),
        "synthetic_negative_passes": sum(not repetition_collapse_v2(value) for value in SYNTHETIC_NEGATIVES),
        "synthetic_negative_observations": len(SYNTHETIC_NEGATIVES),
        "referenced_false_positive_corrections": sum(not row["v2"] for row in referenced_negatives),
        "referenced_false_positive_observations": len(referenced_negatives),
        "actual_v443_loop_detected": repetition_collapse_v2(str(actual_positive["output"])),
        "actual_v443_loop_probe_id": positive_probe_id,
        "actual_v443_loop_output_sha256": hashlib.sha256(
            str(actual_positive["output"]).encode("utf-8")
        ).hexdigest(),
    }
    gates = {
        "all_synthetic_positives_detected": controls["synthetic_positive_passes"]
        == controls["synthetic_positive_observations"],
        "all_synthetic_negatives_rejected": controls["synthetic_negative_passes"]
        == controls["synthetic_negative_observations"],
        "all_referenced_false_positives_corrected": controls["referenced_false_positive_corrections"]
        == controls["referenced_false_positive_observations"],
        "actual_v443_loop_retained": controls["actual_v443_loop_detected"],
        "teacher_zero_collapse": systems["teacher_T0"]["v2_collapses"] == 0,
        "failed_A3_sensitivity": systems["failed_A3"]["v2_collapses"] >= 1000,
        "failed_C3_sensitivity": systems["failed_C3"]["v2_collapses"] >= 1000,
        "failed_D2_sensitivity": systems["failed_D2"]["v2_collapses"] >= 150,
        "historical_v1_recomputed_exactly": all(
            value["historical_v1_collapses"] == value["recomputed_v1_collapses"]
            for value in systems.values()
        ),
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-repetition-metric-audit/1",
        "status": "PASS_V2_CONSTRUCT_AUDIT" if passed else "FAIL_V2_CONSTRUCT_AUDIT",
        "protocol_sha256": _sha256(protocol),
        "systems": systems,
        "controls": controls,
        "referenced_negatives": referenced_negatives,
        "gates": gates,
        "audit_pass": passed,
        "training_performed": False,
        "model_loaded": False,
        "artifact_mutated": False,
        "final_test_accessed": False,
        "phase3_certified": False,
        "claim_boundary": "Read-only development and synthetic construct audit. V1 evidence remains authoritative historically; this audit cannot promote V443 or certify Phase 3.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("ABI_CAPABILITY_COMPILER_PHASE3_REPETITION_METRIC_AUDIT_PROTOCOL_V446.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/abi_capability_compiler_phase3_repetition_metric_audit/audit_v447/result.json"),
    )
    args = parser.parse_args()
    result = run(args.root, args.root / args.protocol, args.root / args.output)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
