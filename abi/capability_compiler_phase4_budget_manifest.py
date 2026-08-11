"""Build the preregistered nested consumed-information budgets for Phase 4."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from . import capability_compiler_phase4_lineage_audit as lineage
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-budget-manifest/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_DETERMINISTIC_NESTED_BUDGET_CONSTRUCTION"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("Phase 4 budget governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 4 budget binding changed: {relative}")
    return protocol, sha256_file(path)


def _rank(rows: list[dict[str, Any]], *, artifact: str, salt: str, groups: tuple[str, ...]) -> list[dict[str, Any]]:
    def group(row: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(str(row[name]) for name in groups)
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group(row)].append(row)
    ordered = []
    for key in sorted(grouped):
        values = sorted(
            grouped[key],
            key=lambda row: hashlib.sha256(
                f"{salt}:{artifact}:{':'.join(key)}:{row['source_attempt_sha256']}".encode("ascii")
            ).hexdigest(),
        )
        ordered.extend(values)
    return ordered


def _prefix_by_group(rows: list[dict[str, Any]], groups: tuple[str, ...], depth: int) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, ...]] = Counter()
    selected = []
    for row in rows:
        key = tuple(str(row[name]) for name in groups)
        if counts[key] < depth:
            selected.append(row)
            counts[key] += 1
    if not counts or set(counts.values()) != {depth}:
        raise Phase3Error("nested budget stratum depth changed")
    return selected


def _record_id(row: Mapping[str, Any]) -> str:
    return str(row.get("ir_record_id", row.get("record_id")))


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    specs = {str(item["id"]): item for item in protocol["teacher_artifacts"]}
    rows = {key: lineage._archive(root, value)[1] for key, value in specs.items()}
    weak = set(protocol["weak_capabilities"])
    rows["v138_targeted_ir"] = [row for row in rows["v138_targeted_ir"] if str(row["capability"]) in weak]

    ranked = {
        "phase1_ir": _rank(rows["phase1_ir"], artifact="phase1_ir", salt=protocol["selection_salt"], groups=("capability",)),
        "v138_targeted_ir": _rank(rows["v138_targeted_ir"], artifact="v138_targeted_ir", salt=protocol["selection_salt"], groups=("capability",)),
        "v480_host_supervision": _rank(rows["v480_host_supervision"], artifact="v480_host_supervision", salt=protocol["selection_salt"], groups=("capability", "builder")),
    }
    ordered_ids = {key: [_record_id(row) for row in value] for key, value in ranked.items()}
    if any(len(value) != len(set(value)) for value in ordered_ids.values()):
        raise Phase3Error("duplicate record identifier in ranked source")

    budgets = []
    previous: dict[str, set[str]] = {key: set() for key in ranked}
    for budget in protocol["budgets"]:
        selected = {
            "phase1_ir": _prefix_by_group(ranked["phase1_ir"], ("capability",), int(budget["phase1_per_capability"])),
            "v138_targeted_ir": _prefix_by_group(ranked["v138_targeted_ir"], ("capability",), int(budget["targeted_per_weak_capability"])),
            "v480_host_supervision": _prefix_by_group(ranked["v480_host_supervision"], ("capability", "builder"), int(budget["host_per_capability_builder"])),
        }
        selected_ids = {key: {_record_id(row) for row in value} for key, value in selected.items()}
        if any(not previous[key].issubset(selected_ids[key]) for key in selected):
            raise Phase3Error("budget nesting failed")
        previous = selected_ids
        union: dict[str, dict[str, Any]] = {}
        memberships = 0
        for artifact in ("phase1_ir", "v138_targeted_ir", "v480_host_supervision"):
            for row in selected[artifact]:
                attempt = str(row["source_attempt_sha256"])
                token_count = int(row.get("authoritative_teacher_tokens", row.get("source_teacher_output_tokens", 0)))
                current = union.setdefault(attempt, {"teacher_tokens": token_count, "artifacts": []})
                if current["teacher_tokens"] != token_count:
                    raise Phase3Error("teacher-token count changed across duplicate source attempt")
                current["artifacts"].append(artifact)
                memberships += 1
        budget_result = {
            "id": str(budget["id"]),
            "fraction": float(budget["fraction"]),
            "record_memberships": memberships,
            "unique_source_attempts": len(union),
            "authoritative_teacher_output_tokens": sum(int(value["teacher_tokens"]) for value in union.values()),
            "duplicate_memberships": memberships - len(union),
            "records": {key: len(value) for key, value in selected.items()},
            "selection_sha256": hashlib.sha256(canonical_json_bytes({key: sorted(value) for key, value in selected_ids.items()})).hexdigest(),
        }
        budgets.append(budget_result)

    gates = {
        "five_preregistered_budgets": len(budgets) == 5,
        "strictly_increasing_unique_attempts": all(a["unique_source_attempts"] < b["unique_source_attempts"] for a, b in zip(budgets, budgets[1:])),
        "full_budget_matches_consumption_audit_attempts": budgets[-1]["unique_source_attempts"] == 9596,
        "full_budget_matches_consumption_audit_tokens": budgets[-1]["authoritative_teacher_output_tokens"] == 294212,
        "teacher_model_not_loaded": True,
        "neural_training_not_performed": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-budget-manifest-result/1",
        "status": "PASS_NESTED_BUDGET_MANIFEST_TRAINING_PROTOCOL_MAY_BE_SEALED" if all(gates.values()) else "FAIL_CLOSED_BUDGET_MANIFEST",
        "protocol_sha256": protocol_sha,
        "selection_salt": protocol["selection_salt"],
        "selection_method": "within-stratum ascending SHA256(salt:artifact:stratum:source_attempt_sha256), then prefix",
        "ordered_record_ids": ordered_ids,
        "ordered_record_ids_sha256": hashlib.sha256(canonical_json_bytes(ordered_ids)).hexdigest(),
        "budgets": budgets,
        "gates": gates,
        "adaptive_order": protocol["adaptive_order"],
        "seed_policy": protocol["seed_policy"],
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Deterministic nested consumed-information budget construction only; no training, frontier result, minimum, Phase 4 certificate, final-test result, or superiority claim.",
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
        raise Phase3Error(f"immutable budget output exists: {output}")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "ordered_record_ids"}, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
