"""Read-only failure taxonomy for the frozen V693 metamorphic outputs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-metamorphic-failure-taxonomy/1"
_CONSECUTIVE_SURFACE_LOOP = re.compile(r"([A-Za-z]{3,32})\1{3,}", re.IGNORECASE)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(not isinstance(row, dict) for row in rows):
        raise Phase3Error(f"expected JSONL objects: {path}")
    return rows


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_FROZEN_OUTPUT_TAXONOMY"
        or protocol.get("new_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("candidate_construction_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("failure-taxonomy governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"failure-taxonomy binding changed: {relative}")
    return protocol, sha256_file(path)


def classify(output: str, evaluator: Mapping[str, Any]) -> dict[str, Any]:
    labels = [str(value) for value in evaluator.get("values", [])]
    if evaluator.get("kind") != "ordered_contains" or len(labels) != 3:
        raise Phase3Error("taxonomy requires a three-label ordered evaluator")
    positions = [output.find(label) for label in labels]
    exact_count = sum(position >= 0 for position in positions)
    all_exact = exact_count == len(labels)
    exact_ordered = all_exact and positions == sorted(positions)
    code = labels[0].rsplit("-", 1)[0]
    suffixes = [label.rsplit("-", 1)[1] for label in labels]
    suffix_count = sum(f"-{suffix}]" in output for suffix in suffixes)
    surface_loop = bool(_CONSECUTIVE_SURFACE_LOOP.search(output))
    if exact_ordered:
        primary = "pass"
    elif all_exact:
        primary = "complete_exact_labels_wrong_order"
    elif code not in output:
        primary = "identifier_stem_absent"
    else:
        primary = "partial_exact_identifier_copy"
    return {
        "primary": primary,
        "exact_label_count": exact_count,
        "all_exact_labels_present": all_exact,
        "exact_labels_ordered": exact_ordered,
        "identifier_stem_present": code in output,
        "expected_suffix_count": suffix_count,
        "surface_loop_suspected": surface_loop,
    }


def _load_bound_rows(root: Path, protocol: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    suite = _jsonl(root / protocol["suite"])
    outputs = _jsonl(root / protocol["outputs"])
    expected = int(protocol["expected_records_per_system"])
    if len(suite) != expected or len(outputs) != expected * 2:
        raise Phase3Error("frozen taxonomy depth changed")
    identifiers = {str(row["ir_record_id"]) for row in suite}
    for system in protocol["systems"]:
        system_rows = [row for row in outputs if row["system"] == system]
        if len(system_rows) != expected or {str(row["record_id"]) for row in system_rows} != identifiers:
            raise Phase3Error(f"frozen output pairing changed: {system}")
    return suite, outputs


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    suite, outputs = _load_bound_rows(root, protocol)
    return {
        "status": "PASS_FROZEN_OUTPUT_TAXONOMY_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "suite_records": len(suite),
        "output_records": len(outputs),
        "systems": protocol["systems"],
        "paired_complete": True,
        "new_inference_performed": False,
        "training_performed": False,
        "candidate_constructed": False,
        "final_test_accessed": False,
    }


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable taxonomy output already exists")
    suite, outputs = _load_bound_rows(root, protocol)
    by_id = {str(row["ir_record_id"]): row for row in suite}
    evidence = []
    systems: dict[str, Any] = {}
    for system in protocol["systems"]:
        rows = [row for row in outputs if row["system"] == system]
        classified = []
        for row in rows:
            source = by_id[str(row["record_id"])]
            taxonomy = classify(str(row["output"]), source["functional_evaluator"])
            item = {"record_id": row["record_id"], "system": system, "namespace": source["namespace"], "family": source["family"], **taxonomy}
            evidence.append(item)
            classified.append(item)
        counts = Counter(row["primary"] for row in classified)
        failures = len(classified) - counts["pass"]
        copy_failures = counts["identifier_stem_absent"] + counts["partial_exact_identifier_copy"]
        all_exact = sum(row["all_exact_labels_present"] for row in classified)
        ordered = sum(row["exact_labels_ordered"] for row in classified)
        systems[system] = {
            "observations": len(classified),
            "primary_counts": dict(sorted(counts.items())),
            "failures": failures,
            "copy_failures": copy_failures,
            "copy_failure_share_of_failures": copy_failures / failures if failures else 0.0,
            "all_exact_label_rows": all_exact,
            "ordering_success_given_all_exact": ordered / all_exact if all_exact else 0.0,
            "mean_exact_label_recall": sum(row["exact_label_count"] for row in classified) / (3 * len(classified)),
            "identifier_stem_present": sum(row["identifier_stem_present"] for row in classified),
            "surface_loops_suspected": sum(row["surface_loop_suspected"] for row in classified),
        }
    adapted = systems["adapted"]
    gates = {
        "adapted_copy_failure_share": adapted["copy_failure_share_of_failures"] >= float(protocol["attribution_thresholds"]["copy_failure_share_minimum"]),
        "adapted_order_given_complete": adapted["ordering_success_given_all_exact"] >= float(protocol["attribution_thresholds"]["ordering_given_complete_minimum"]),
        "adapted_has_complete_rows": adapted["all_exact_label_rows"] >= int(protocol["attribution_thresholds"]["minimum_complete_rows"]),
        "paired_inventory_complete": len(evidence) == len(outputs),
        "no_new_inference": True,
        "no_training": True,
        "no_candidate": True,
        "final_test_not_accessed": True,
    }
    status = "PASS_IDENTIFIER_COPY_BOTTLENECK_LOCALIZED" if all(gates.values()) else "COMPLETE_MIXED_OR_ORDERING_FAILURE_TAXONOMY"
    raw = output.parent / "taxonomy_rows.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in evidence))
    result = {
        "format": "abi-capability-compiler-phase4-metamorphic-failure-taxonomy-result/1",
        "status": status,
        "protocol_sha256": protocol_sha,
        "systems": systems,
        "gates": gates,
        "taxonomy_rows_sha256": sha256_file(raw),
        "new_inference_performed": False,
        "training_performed": False,
        "candidate_constructed": False,
        "final_test_accessed": False,
        "interpretation": "A copy-bottleneck pass supports protocol design for at most one prospective generic prompt-identifier pointer/copy mechanism. It does not authorize implementation, training, candidate construction, or promotion.",
        "claim_boundary": "Frozen-output error attribution only; no repaired architecture, candidate, stable frontier, matched baseline, final test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("audit")
    command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else audit(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
