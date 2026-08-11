"""Exact locked-V2 incremental contract-guard feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_audit import _contains_any_values
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_repetition_v2 import TOKEN_PATTERN, repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-contract-guard-v2-audit/1"
WEAK = ("abstention", "coherence", "fluent_realization", "tone_control")


def _json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_EXACT_V2_CONTRACT_GUARD_FEASIBILITY" or protocol.get("neural_training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("exact V2 guard governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected: raise Phase3Error(f"exact V2 guard binding changed: {relative}")
    return protocol, sha256_file(path)


def truncate_at_first_v2_collapse(output: str) -> tuple[str, bool]:
    if not repetition_collapse_v2(output): return output, False
    for match in TOKEN_PATTERN.finditer(output):
        prefix = output[: match.end()]
        if repetition_collapse_v2(prefix):
            value = output[: match.start()].rstrip()
            if repetition_collapse_v2(value): raise Phase3Error("exact V2 prefix guard failed to remove collapse")
            return value, True
    raise Phase3Error("collapsed output has no detectable V2 transition")


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists(): raise Phase3Error(f"immutable exact V2 guard output exists: {output}")
    with zipfile.ZipFile(root / protocol["artifact"]["path"], "r") as archive: artifact = [json.loads(line) for line in archive.read("records.jsonl").splitlines() if line]
    marker_sets = [set(_contains_any_values(row["functional_evaluator"])) for row in artifact if row["capability"] == "abstention" and _contains_any_values(row["functional_evaluator"])]
    markers = tuple(sorted(set.intersection(*marker_sets))); marker = str(protocol["guard"]["canonical_abstention_marker"]); clause = str(protocol["guard"]["canonical_abstention_clause"])
    if marker not in markers or marker.casefold() not in clause.casefold(): raise Phase3Error("exact V2 abstention clause lost artifact provenance")
    catalog = _json(root / protocol["development"]["catalog"]); probes = {str(row["probe_id"]): row for row in catalog["probes"] if row.get("split") == "validation"}; source = [json.loads(line) for line in (root / protocol["development"]["candidate_outputs"]).read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for row in source:
        original = str(row["output"]); capability = str(row["capability"]); value, truncated = truncate_at_first_v2_collapse(original)
        prefixed = False
        if capability == "abstention" and not any(item.casefold() in value.casefold() for item in markers): value = clause + (" " + value if value else ""); prefixed = True
        probe = probes[str(row["probe_id"])]
        rows.append({"probe_id": str(row["probe_id"]), "capability": capability, "original_output": original, "guarded_output": value, "v2_truncated": truncated, "abstention_clause_prefixed": prefixed, "original_functional_v1": evaluate_functional(original, probe["evaluator"]), "guarded_functional_v1": evaluate_functional(value, probe["evaluator"]), "guarded_functional_v2": evaluate_functional_v2(value, probe["evaluator"], capability), "original_collapse_v2": repetition_collapse_v2(original), "guarded_collapse_v2": repetition_collapse_v2(value)})
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]; passes = sum(row["guarded_functional_v1"] for row in values); per[capability] = {"passes_v1": passes, "observations": len(values), "wilson_v1": wilson(passes, len(values)), "collapses_v2": sum(row["guarded_collapse_v2"] for row in values)}
    existing_lost = sum(row["original_functional_v1"] and not row["guarded_functional_v1"] for row in rows); noncollapsed_changed = sum(not row["original_collapse_v2"] and not row["abstention_clause_prefixed"] and row["original_output"] != row["guarded_output"] for row in rows); strong_changed = sum(row["capability"] not in WEAK and row["original_output"] != row["guarded_output"] for row in rows); collapses = sum(row["guarded_collapse_v2"] for row in rows)
    gates_cfg = protocol["gates"]; gates = {"existing_passes_lost_zero": existing_lost == 0, "noncollapsed_outputs_changed_zero": noncollapsed_changed == 0, "strong_outputs_changed_zero": strong_changed == 0, "repetition_collapses_zero": collapses == 0, "critical_abstention": per["abstention"]["wilson_v1"]["point"] >= float(gates_cfg["critical_point_minimum"]) and per["abstention"]["wilson_v1"]["lower_95"] >= float(gates_cfg["critical_wilson_lower_minimum"]), "per_capability_quality": all(value["wilson_v1"]["point"] >= float(gates_cfg["per_capability_point_minimum"]) and value["wilson_v1"]["lower_95"] >= float(gates_cfg["per_capability_wilson_lower_minimum"]) for value in per.values()), "final_test_not_accessed": True}
    raw = output.parent / "guarded_outputs.jsonl"; raw.parent.mkdir(parents=True, exist_ok=True); _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in rows)); passed = all(gates.values())
    result = {"format": FORMAT, "status": "PASS_EXACT_V2_CONTRACT_GUARD_FEASIBILITY" if passed else "FAIL_EXACT_V2_CONTRACT_GUARD_CLOSED", "protocol_sha256": protocol_sha, "artifact_derived_markers": list(markers), "canonical_abstention_clause": clause, "observations": len(rows), "functional_v1_passes": sum(row["guarded_functional_v1"] for row in rows), "repetition_collapses_v2": collapses, "v2_truncations": sum(row["v2_truncated"] for row in rows), "abstention_prefixes": sum(row["abstention_clause_prefixed"] for row in rows), "existing_functional_passes_lost": existing_lost, "noncollapsed_outputs_changed": noncollapsed_changed, "strong_outputs_changed": strong_changed, "per_capability": per, "gates": gates, "raw_outputs_sha256": sha256_file(raw), "neural_training_performed": False, "artifact_mutated": False, "final_test_accessed": False, "phase3_certified": False}; result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest(); _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n"); return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_CONTRACT_GUARD_V2_AUDIT_PROTOCOL_V491.json"); parser.add_argument("--output", default="results/abi_capability_compiler_phase3_contract_guard_v2/audit_v492/result.json"); args = parser.parse_args(argv); root = Path.cwd().resolve(); result = run(root, root / args.protocol, root / args.output); print(json.dumps({"status": result["status"], "functional_v1_passes": result["functional_v1_passes"], "repetition_collapses_v2": result["repetition_collapses_v2"], "abstention": result["per_capability"]["abstention"], "gates": result["gates"], "evidence_sha256": result["evidence_sha256"]}, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
