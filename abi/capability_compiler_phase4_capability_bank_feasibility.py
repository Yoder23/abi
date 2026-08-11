"""Cross-validated read-only feasibility audit for capability-isolated execution."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_final_controls import wilson
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap


FORMAT = "abi-capability-compiler-phase4-capability-bank-feasibility/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _fold(probe_id: str, folds: int) -> int:
    return int.from_bytes(hashlib.sha256(probe_id.encode()).digest()[:8], "big") % folds


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_CROSS_VALIDATED_FEASIBILITY"
        or protocol.get("training_authorized") is not False
        or protocol.get("promotion_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("capability-bank feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"capability-bank feasibility binding changed: {relative}")
    return protocol, sha256_file(path)


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable capability-bank feasibility output exists")
    systems = {name: {row["probe_id"]: row for row in _rows(root / path)} for name, path in protocol["systems"].items()}
    ids = {name: set(rows) for name, rows in systems.items()}
    if len({frozenset(value) for value in ids.values()}) != 1 or len(next(iter(ids.values()))) != 1400:
        raise Phase3Error("capability-bank populations differ")
    probes = {row["probe_id"]: row for row in development_probes(root / protocol["catalog"])}
    teacher = {row["probe_id"]: row for row in _rows(root / protocol["teacher_reference"])}
    folds = int(protocol["folds"])
    selected_rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    ordered = list(protocol["tie_break_order"])
    for capability in CAPABILITIES:
        capability_ids = sorted(probe_id for probe_id, probe in probes.items() if probe["canonical_capability"] == capability)
        for heldout in range(folds):
            training_ids = [probe_id for probe_id in capability_ids if _fold(probe_id, folds) != heldout]
            scores = {name: sum(bool(systems[name][probe_id]["functional_pass_v1"]) for probe_id in training_ids) for name in ordered}
            best_score = max(scores.values())
            selected = next(name for name in ordered if scores[name] == best_score)
            heldout_ids = [probe_id for probe_id in capability_ids if _fold(probe_id, folds) == heldout]
            selections.append({"capability": capability, "fold": heldout, "selected_system": selected, "selection_passes": best_score, "selection_observations": len(training_ids), "heldout_observations": len(heldout_ids)})
            for probe_id in heldout_ids:
                row = systems[selected][probe_id]
                selected_rows.append({"probe_id": probe_id, "capability": capability, "selected_system": selected, "functional_pass_v1": bool(row["functional_pass_v1"]), "repetition_collapse_v2": bool(row["repetition_collapse_v2"]), "output": row["output"]})
    if len(selected_rows) != 1400 or len({row["probe_id"] for row in selected_rows}) != 1400:
        raise Phase3Error("cross-validated selection coverage changed")
    per = {}
    for capability in CAPABILITIES:
        rows = [row for row in selected_rows if row["capability"] == capability]
        passed = sum(row["functional_pass_v1"] for row in rows)
        per[capability] = {"passes": passed, "observations": len(rows), "wilson": wilson(passed, len(rows)), "collapses": sum(row["repetition_collapse_v2"] for row in rows)}
    paired = [{"capability": row["capability"], "candidate_pass": row["functional_pass_v1"], "teacher_pass": evaluate_functional(str(teacher[row["probe_id"]]["output"]), probes[row["probe_id"]]["evaluator"])} for row in selected_rows]
    relative = paired_stratified_bootstrap(paired, replicates=10_000, seed=int(protocol["bootstrap_seed"]))
    thresholds = protocol["thresholds"]
    gates = {
        "per_capability": all(value["wilson"]["point"] >= thresholds["per_capability_point"] and value["wilson"]["lower_95"] >= thresholds["per_capability_lower"] for value in per.values()),
        "critical": all(per[name]["wilson"]["point"] >= thresholds["critical_point"] and per[name]["wilson"]["lower_95"] >= thresholds["critical_lower"] for name in protocol["critical_capabilities"]),
        "zero_collapse": sum(row["repetition_collapse_v2"] for row in selected_rows) == 0,
        "teacher_noninferior": relative["lower_95"] >= thresholds["teacher_relative_lower"],
        "cross_validated": True,
        "no_training": True,
        "no_promotion": True,
        "final_test_not_accessed": True,
    }
    raw = output.parent / "selected_outputs.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in selected_rows))
    result = {
        "format": "abi-capability-compiler-phase4-capability-bank-feasibility-result/1",
        "status": "PASS_CAPABILITY_ISOLATION_FEASIBLE_ARCHITECTURE_DESIGN_SUPPORTED" if all(gates.values()) else "FAIL_CAPABILITY_ISOLATION_NOT_SUPPORTED",
        "protocol_sha256": protocol_sha,
        "systems": ordered,
        "folds": folds,
        "selections": selections,
        "selection_counts": dict(Counter(row["selected_system"] for row in selections)),
        "functional_passes": sum(row["functional_pass_v1"] for row in selected_rows),
        "observations": len(selected_rows),
        "per_capability": per,
        "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in selected_rows),
        "teacher_comparison": relative,
        "gates": gates,
        "raw_outputs_sha256": sha256_file(raw),
        "training_performed": False,
        "promotion_authorized": False,
        "final_test_accessed": False,
        "interpretation": "Cross-validated development feasibility only. A pass supports one sparse capability-isolated adaptation design; it does not authorize packaging the post-hoc checkpoint bank or claim out-of-sample quality.",
        "claim_boundary": "Read-only development diagnostic only; no candidate, stable frontier, minimum, matched baseline, final test, Phase 4 certificate, or superiority claim."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = audit(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
