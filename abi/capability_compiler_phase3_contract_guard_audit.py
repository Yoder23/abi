"""Read-only artifact-derived capability-contract guard feasibility audit."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
import zipfile

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-contract-guard-audit/1"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_ARTIFACT_DERIVED_CONTRACT_GUARD_FEASIBILITY"
        or protocol.get("neural_training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("contract guard governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"contract guard binding changed: {relative}")
    return protocol, sha256_file(path)


def truncate_before_fifth_repeated_gram(output: str) -> tuple[str, bool]:
    matches = list(re.finditer(r"\S+", output))
    words: list[str] = []
    counts: Counter[tuple[str, ...]] = Counter()
    for match in matches:
        words.append(match.group(0).casefold())
        would_collapse = False
        grams = []
        for width in (1, 2, 3, 4):
            if len(words) >= width:
                gram = tuple(words[-width:]); grams.append(gram)
                if counts[gram] + 1 >= 5:
                    would_collapse = True
        if would_collapse:
            value = output[: match.start()].rstrip()
            return value, value != output
        for gram in grams:
            counts[gram] += 1
    return output, False


def apply_contract_guard(output: str, capability: str, markers: tuple[str, ...], canonical_clause: str) -> tuple[str, dict[str, bool]]:
    value, truncated = truncate_before_fifth_repeated_gram(output)
    abstention_prefixed = False
    if capability == "abstention" and not any(marker.casefold() in value.casefold() for marker in markers):
        value = canonical_clause + (" " + value if value else "")
        abstention_prefixed = True
    return value, {"repetition_truncated": truncated, "abstention_clause_prefixed": abstention_prefixed}


def _contains_any_values(evaluator: Mapping[str, Any]) -> list[str]:
    kind = evaluator["kind"]
    if kind == "contains_any":
        return [str(value) for value in evaluator["values"]]
    if kind == "all_of":
        return [value for rule in evaluator["rules"] for value in _contains_any_values(rule)]
    return []


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable contract guard output exists: {output}")
    with zipfile.ZipFile(root / protocol["artifact"]["path"], "r") as archive:
        artifact_rows = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]
    marker_sets = []
    for row in artifact_rows:
        if row["capability"] == "abstention":
            values = tuple(sorted(set(_contains_any_values(row["functional_evaluator"]))))
            if values:
                marker_sets.append(set(values))
    markers = tuple(sorted(set.intersection(*marker_sets)))
    required_marker = str(protocol["guard"]["canonical_abstention_marker"])
    canonical_clause = str(protocol["guard"]["canonical_abstention_clause"])
    if required_marker not in markers or required_marker.casefold() not in canonical_clause.casefold():
        raise Phase3Error("canonical abstention clause is not artifact-derived")
    catalog = _json(root / protocol["development"]["catalog"])
    probes = {str(row["probe_id"]): row for row in catalog["probes"] if row.get("split") == "validation"}
    rows = [json.loads(line) for line in (root / protocol["development"]["candidate_outputs"]).read_text(encoding="utf-8").splitlines() if line]
    guarded = []
    for row in rows:
        probe = probes[str(row["probe_id"])]
        value, changes = apply_contract_guard(str(row["output"]), str(row["capability"]), markers, canonical_clause)
        guarded.append({
            "probe_id": str(row["probe_id"]), "capability": str(row["capability"]), "original_output": str(row["output"]), "guarded_output": value,
            **changes,
            "original_functional_v1": evaluate_functional(str(row["output"]), probe["evaluator"]),
            "guarded_functional_v1": evaluate_functional(value, probe["evaluator"]),
            "guarded_functional_v2": evaluate_functional_v2(value, probe["evaluator"], str(row["capability"])),
            "original_collapse_v2": repetition_collapse_v2(str(row["output"])), "guarded_collapse_v2": repetition_collapse_v2(value),
        })
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in guarded if row["capability"] == capability]; passes = sum(row["guarded_functional_v1"] for row in values); per[capability] = {"passes_v1": passes, "observations": len(values), "wilson_v1": wilson(passes, len(values)), "collapses_v2": sum(row["guarded_collapse_v2"] for row in values)}
    existing_passes_lost = sum(row["original_functional_v1"] and not row["guarded_functional_v1"] for row in guarded)
    strong_changed = sum(row["capability"] not in ("abstention", "coherence", "fluent_realization", "tone_control") and row["original_output"] != row["guarded_output"] for row in guarded)
    gates = {
        "existing_functional_passes_lost_zero": existing_passes_lost == 0,
        "strong_outputs_changed_zero": strong_changed == 0,
        "repetition_collapses_zero": sum(row["guarded_collapse_v2"] for row in guarded) == 0,
        "critical_abstention_point": per["abstention"]["wilson_v1"]["point"] >= float(protocol["gates"]["critical_point_minimum"]),
        "critical_abstention_wilson": per["abstention"]["wilson_v1"]["lower_95"] >= float(protocol["gates"]["critical_wilson_lower_minimum"]),
        "per_capability_quality": all(value["wilson_v1"]["point"] >= float(protocol["gates"]["per_capability_point_minimum"]) and value["wilson_v1"]["lower_95"] >= float(protocol["gates"]["per_capability_wilson_lower_minimum"]) for value in per.values()),
        "final_test_not_accessed": True,
    }
    raw = output.parent / "guarded_outputs.jsonl"; raw.parent.mkdir(parents=True, exist_ok=True); _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in guarded))
    passed = all(gates.values())
    result = {
        "format": FORMAT, "status": "PASS_ARTIFACT_DERIVED_CONTRACT_GUARD_FEASIBILITY" if passed else "FAIL_CONTRACT_GUARD_FEASIBILITY_CLOSED", "protocol_sha256": protocol_sha,
        "artifact_derived_markers": list(markers), "canonical_abstention_clause": canonical_clause, "observations": len(guarded), "functional_v1_passes": sum(row["guarded_functional_v1"] for row in guarded), "repetition_collapses_v2": sum(row["guarded_collapse_v2"] for row in guarded), "repetition_truncations": sum(row["repetition_truncated"] for row in guarded), "abstention_prefixes": sum(row["abstention_clause_prefixed"] for row in guarded), "existing_functional_passes_lost": existing_passes_lost, "strong_outputs_changed": strong_changed, "per_capability": per, "gates": gates, "raw_outputs_sha256": sha256_file(raw), "neural_training_performed": False, "artifact_mutated": False, "final_test_accessed": False, "phase3_certified": False,
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CONTRACT_GUARD_AUDIT_PROTOCOL_V489.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_contract_guard/audit_v490/result.json"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output); print(json.dumps({"status": result["status"], "functional_v1_passes": result["functional_v1_passes"], "repetition_collapses_v2": result["repetition_collapses_v2"], "abstention": result["per_capability"]["abstention"], "gates": result["gates"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
