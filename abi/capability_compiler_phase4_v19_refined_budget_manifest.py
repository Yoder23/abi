"""Freeze deterministic B50/B60/B70 refinements inside the verified B40-B80 bracket."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES
from .capability_compiler_phase4_budget_manifest import (
    _prefix_by_group,
    _rank,
    _record_id,
)
from .capability_compiler_phase4_lineage_audit import _archive


FORMAT = "abi-capability-compiler-phase4-v19-refined-budget-manifest/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_NESTED_B40_B80_INTERVAL_REFINEMENT"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("refined budget manifest governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"refined budget manifest binding changed: {relative}")
    return protocol, sha256_file(path)


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    base_protocol = _json(root / protocol["base_lineage_protocol"])
    parent_manifest = _json(root / protocol["parent_manifest"])
    source_specs = {str(item["id"]): item for item in base_protocol["teacher_artifacts"]}
    source_rows = {key: _archive(root, value)[1] for key, value in source_specs.items()}
    source_rows["v138_targeted_ir"] = [
        row for row in source_rows["v138_targeted_ir"]
        if str(row["capability"]) in set(WEAK_CAPABILITIES)
    ]
    ranked = {
        "phase1_ir": _rank(source_rows["phase1_ir"], artifact="phase1_ir", salt=parent_manifest["selection_salt"], groups=("capability",)),
        "v138_targeted_ir": _rank(source_rows["v138_targeted_ir"], artifact="v138_targeted_ir", salt=parent_manifest["selection_salt"], groups=("capability",)),
        "v480_host_supervision": _rank(source_rows["v480_host_supervision"], artifact="v480_host_supervision", salt=parent_manifest["selection_salt"], groups=("capability", "builder")),
    }
    manifests = []
    selected_sets: dict[str, dict[str, set[str]]] = {}
    previous: dict[str, set[str]] | None = None
    for budget in protocol["budgets"]:
        budget_id = str(budget["id"])
        # Reuse the exact original rank/salt implementation. Calculate the
        # prospective identity first, then supply it to the unchanged selector.
        selected = {
            "phase1_ir": _prefix_by_group(ranked["phase1_ir"], ("capability",), int(budget["phase1_per_capability"])),
            "v138_targeted_ir": _prefix_by_group(ranked["v138_targeted_ir"], ("capability",), int(budget["targeted_per_weak_capability"])),
            "v480_host_supervision": _prefix_by_group(ranked["v480_host_supervision"], ("capability", "builder"), int(budget["host_per_capability_builder"])),
        }
        ids = {key: {_record_id(row) for row in values} for key, values in selected.items()}
        if previous is not None and any(not previous[key].issubset(ids[key]) for key in ids):
            raise Phase3Error("refined budget nesting failed")
        previous = ids
        selected_sets[budget_id] = ids
        union: dict[str, dict[str, Any]] = {}
        memberships = 0
        for artifact, rows in selected.items():
            for row in rows:
                attempt = str(row["source_attempt_sha256"])
                tokens = int(row.get("authoritative_teacher_tokens", row.get("source_teacher_output_tokens", 0)))
                item = union.setdefault(attempt, {"teacher_tokens": tokens, "artifacts": []})
                if item["teacher_tokens"] != tokens:
                    raise Phase3Error("teacher-token count changed across duplicate attempt")
                item["artifacts"].append(artifact)
                memberships += 1
        manifests.append({
            "id": budget_id,
            "fraction": float(budget["fraction"]),
            "record_memberships": memberships,
            "unique_source_attempts": len(union),
            "authoritative_teacher_output_tokens": sum(int(value["teacher_tokens"]) for value in union.values()),
            "duplicate_memberships": memberships - len(union),
            "records": {key: len(values) for key, values in selected.items()},
            "selection_sha256": hashlib.sha256(canonical_json_bytes({key: sorted(value) for key, value in ids.items()})).hexdigest(),
        })
    parent = {str(row["id"]): row for row in parent_manifest["budgets"]}
    gates = {
        "five_nested_budgets": [row["id"] for row in manifests] == ["B40", "B50", "B60", "B70", "B80"],
        "b40_identity_preserved": manifests[0]["selection_sha256"] == parent["B40"]["selection_sha256"],
        "b80_identity_preserved": manifests[-1]["selection_sha256"] == parent["B80"]["selection_sha256"],
        "strictly_increasing_attempts": all(left["unique_source_attempts"] < right["unique_source_attempts"] for left, right in zip(manifests, manifests[1:])),
        "strictly_increasing_tokens": all(left["authoritative_teacher_output_tokens"] < right["authoritative_teacher_output_tokens"] for left, right in zip(manifests, manifests[1:])),
        "all_selection_hashes_unique": len({row["selection_sha256"] for row in manifests}) == 5,
        "teacher_model_not_loaded": True,
        "training_not_performed": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-v19-refined-budget-manifest-result/1",
        "status": "PASS_REFINED_B40_B80_NESTED_BUDGET_MANIFEST" if all(gates.values()) else "FAIL_REFINED_BUDGET_MANIFEST",
        "protocol_sha256": protocol_sha,
        "selection_salt": parent_manifest["selection_salt"],
        "selection_method": parent_manifest["selection_method"],
        "parent_ordered_record_ids_sha256": parent_manifest["ordered_record_ids_sha256"],
        "budgets": manifests,
        "gates": gates,
        "adaptive_order": protocol["adaptive_order"],
        "seed_policy": protocol["seed_policy"],
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Refined nested information-budget identities only. No new training result, frontier minimum, matched baseline, final test, Phase 4, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    output = root / args.output
    if output.exists():
        raise Phase3Error(f"immutable refined budget manifest exists: {output}")
    result = run(root, root / args.protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
