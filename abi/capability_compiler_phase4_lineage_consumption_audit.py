"""Repair Phase 4 accounting by distinguishing archived from consumed records."""

from __future__ import annotations

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import capability_compiler_phase4_lineage_audit as base
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-lineage-consumption-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CONSUMED_INFORMATION_REPAIR"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("historical_v558_changed") is not False
    ):
        raise Phase3Error("Phase 4 consumption-audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 4 consumption binding changed: {relative}")
    return protocol, sha256_file(path)


def _selected(rows: list[dict[str, Any]], rule: Mapping[str, Any]) -> list[dict[str, Any]]:
    capabilities = rule.get("capabilities")
    selected = rows if capabilities == "ALL" else [row for row in rows if str(row["capability"]) in set(capabilities)]
    if len(selected) != int(rule["expected_records"]):
        raise Phase3Error(f"consumed record depth changed: {rule['id']}")
    return selected


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    historical = base.run(root, root / protocol["historical_audit"]["protocol"])
    if historical["evidence_sha256"] != protocol["historical_audit"]["evidence_sha256"]:
        raise Phase3Error("historical V558 evidence changed")

    archive_specs = {str(item["id"]): item for item in protocol["teacher_artifacts"]}
    archive_rows: dict[str, list[dict[str, Any]]] = {}
    for artifact_id, spec in archive_specs.items():
        _, archive_rows[artifact_id] = base._archive(root, spec)

    consumed: dict[str, dict[str, Any]] = {}
    union: dict[str, dict[str, Any]] = {}
    for rule in protocol["consumption_rules"]:
        rows = _selected(archive_rows[str(rule["artifact"])], rule)
        attempts = {str(row["source_attempt_sha256"]) for row in rows}
        tokens = sum(int(row.get("authoritative_teacher_tokens", row.get("source_teacher_output_tokens", 0))) for row in rows)
        consumed[str(rule["id"])] = {
            "artifact": str(rule["artifact"]),
            "consumer_stages": list(rule["consumer_stages"]),
            "records": len(rows),
            "unique_source_attempts": len(attempts),
            "authoritative_teacher_output_tokens": tokens,
            "capability_counts": dict(sorted(Counter(str(row["capability"]) for row in rows).items())),
        }
        for row in rows:
            attempt = str(row["source_attempt_sha256"])
            value = {
                "teacher_output_tokens": int(row.get("authoritative_teacher_tokens", row.get("source_teacher_output_tokens", 0))),
                "rules": [],
            }
            current = union.setdefault(attempt, value)
            if current["teacher_output_tokens"] != value["teacher_output_tokens"]:
                raise Phase3Error("same source attempt has inconsistent teacher token accounting")
            current["rules"].append(str(rule["id"]))

    unique_tokens = sum(int(row["teacher_output_tokens"]) for row in union.values())
    duplicate_memberships = sum(max(0, len(row["rules"]) - 1) for row in union.values())
    archived_attempts = int(historical["unique_imported_information"]["source_attempts"])
    gates = {
        "historical_container_audit_still_passes": historical["status"].startswith("PASS"),
        "all_consumption_depths_match": True,
        "unused_targeted_records_excluded_from_consumed_budget": len(archive_rows["v138_targeted_ir"]) - consumed["v474_targeted_weak"]["records"] == 5000,
        "full_phase1_consumption_retained": consumed["phase1_all"]["records"] == 7000,
        "full_v480_consumption_retained": consumed["v480_all"]["records"] == 1280,
        "consumed_less_than_archived": len(union) < archived_attempts,
        "teacher_model_loaded": False,
        "neural_training_performed": False,
        "final_test_accessed": False,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-lineage-consumption-audit-result/1",
        "status": "PASS_CONSUMED_INFORMATION_LINEAGE_FRONTIER_PROTOCOL_OPEN" if passed else "FAIL_CLOSED_CONSUMPTION_LINEAGE",
        "protocol_sha256": protocol_sha,
        "historical_v558": {
            "preserved": True,
            "container_records": sum(value["records"] for value in historical["teacher_artifacts"].values()),
            "container_unique_source_attempts": archived_attempts,
            "container_unique_teacher_output_tokens": int(historical["unique_imported_information"]["authoritative_teacher_output_tokens"]),
            "interpretation": "physical archive inventory, superseded as a consumed-information total",
        },
        "consumption_rules": consumed,
        "consumed_unique_information": {
            "source_attempts": len(union),
            "authoritative_teacher_output_tokens": unique_tokens,
            "duplicate_rule_memberships": duplicate_memberships,
            "stored_logits": 0,
            "stored_hidden_activations": 0,
            "copied_source_parameters": 0,
        },
        "unused_but_archived": {
            "v138_nonweak_records": 5000,
            "counted_in_disk_footprint": True,
            "counted_as_training_information": False,
        },
        "fixed_host_prior": historical["fixed_host_prior"],
        "router": historical["router"],
        "confounds": historical["confounds"],
        "gates": gates,
        "phase2_status_unchanged": historical["phase2_status_unchanged"],
        "phase3_status_unchanged": historical["phase3_status_unchanged"],
        "phase4_certified": False,
        "phase5_open": False,
        "final_test_accessed": False,
        "decision": "Use only consumed records for nested information budgets; separately report full archive disk footprint and the fixed host prior. Rebuild all ABI data-dependent stages from the clean host.",
        "claim_boundary": "Prospective accounting repair only; V558 remains historical. No training, minimum-information, Phase 4 certificate, final-test result, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error(f"immutable audit output exists: {output}")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
