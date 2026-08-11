"""Independent hostile verifier for the route-isolated Phase 3 matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap
from .capability_compiler_repetition_v2 import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase3-route-isolated-verifier/1"
SYSTEMS = ("A0", "A1_label_free", "A2_shuffled", "A3_bridge_only", "A4_monolithic")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _verify_rows(rows: Sequence[Mapping[str, Any]], probes: Mapping[str, Mapping[str, Any]], parent: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row["probe_id"]) for row in rows]
    if len(rows) != 1400 or len(set(ids)) != 1400 or set(ids) != set(probes):
        raise Phase3Error("output prompt identity/depth changed")
    passes = collapses = router = strong = 0
    per = {name: 0 for name in CAPABILITIES}
    for row in rows:
        probe = probes[str(row["probe_id"])]
        capability = str(probe["canonical_capability"])
        output = str(row["output"])
        v1 = evaluate_functional(output, probe["evaluator"])
        v2 = evaluate_functional_v2(output, probe["evaluator"], capability)
        collapse = repetition_collapse_v2(output)
        if str(row["capability"]) != capability or bool(row["functional_pass_v1"]) != v1 or bool(row["functional_pass_v2"]) != v2 or bool(row["repetition_collapse_v2"]) != collapse:
            raise Phase3Error("output semantic annotation mismatch")
        route_ok = str(row["automatic_capability_route"]) == capability
        if bool(row["capability_route_correct"]) != route_ok:
            raise Phase3Error("route annotation mismatch")
        if capability not in {"abstention", "coherence", "fluent_realization", "tone_control"}:
            exact = output == str(parent[str(row["probe_id"])]["output"])
            if bool(row["strong_parent_output_exact"]) != exact:
                raise Phase3Error("strong-parent annotation mismatch")
            strong += int(exact)
        passes += int(v1); per[capability] += int(v1); collapses += int(collapse); router += int(route_ok)
    return {"functional_passes_v1": passes, "per_capability_passes_v1": per, "repetition_collapses_v2": collapses, "router_correct": router, "strong_exact": strong}


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_INDEPENDENT_HOSTILE_RECONSTRUCTION" or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("route-isolated verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"route-isolated verifier binding changed: {relative}")
    if output.exists():
        raise Phase3Error("immutable verifier output exists")
    probes = {str(row["probe_id"]): row for row in development_probes(root / protocol["catalog"])}
    parent = {str(row["probe_id"]): row for row in _load_rows(root / protocol["parent_outputs"])}
    rows = {name: _load_rows(root / path) for name, path in protocol["outputs"].items()}
    reconstructed = {name: _verify_rows(value, probes, parent) for name, value in rows.items()}
    decision = _json(root / protocol["decision"])
    if reconstructed["A0"]["functional_passes_v1"] != int(decision["A0"]["functional_passes_v1"]):
        raise Phase3Error("A0 decision aggregate mismatch")
    paired = {}
    for index, name in enumerate(SYSTEMS[1:]):
        control = {str(row["probe_id"]): row for row in rows[name]}
        values = [{"capability": row["capability"], "candidate_pass": bool(row["functional_pass_v1"]), "teacher_pass": bool(control[str(row["probe_id"])]["functional_pass_v1"])} for row in rows["A0"]]
        comparison = paired_stratified_bootstrap(values, replicates=10_000, seed=5181729 + index)
        recorded = decision["paired_A0_minus_control"][name]
        if any(comparison[key] != recorded[key] for key in ("candidate_minus_teacher", "lower_95", "upper_95", "replicates", "seed")):
            raise Phase3Error("paired decision reconstruction mismatch")
        paired[name] = comparison
    metadata = {name: _json(root / path) for name, path in protocol["metadata"].items()}
    sequences = {value["training"]["record_sequence_sha256"] for value in metadata.values()}
    if len(sequences) != 1 or any(int(value["bridge_parameters"]) != 99840 or int(value["training"]["steps"]) != 2000 or int(value["training"]["observations"]) != 64000 or int(value["training"]["recovery_batches"]) != 476 or value["parent"]["mutated"] is not False for value in metadata.values()):
        raise Phase3Error("training/accounting match changed")
    if int(metadata["A3_bridge_only"]["training"]["teacher_response_tokens_in_loss"]) != 0 or any(int(metadata[name]["training"]["teacher_response_tokens_in_loss"]) <= 0 for name in ("A0", "A1_label_free", "A2_shuffled", "A4_monolithic")):
        raise Phase3Error("teacher-response control accounting changed")

    hostile = {}
    attacks = []
    duplicate = copy.deepcopy(rows["A0"]); duplicate[-1]["probe_id"] = duplicate[0]["probe_id"]; attacks.append(("duplicate_probe", duplicate))
    changed_output = copy.deepcopy(rows["A0"]); changed_output[0]["output"] += " mutation"; attacks.append(("output_mutation", changed_output))
    flipped = copy.deepcopy(rows["A0"]); flipped[0]["functional_pass_v1"] = not flipped[0]["functional_pass_v1"]; attacks.append(("functional_annotation_mutation", flipped))
    changed_route = copy.deepcopy(rows["A0"]); changed_route[0]["automatic_capability_route"] = "tone_control"; attacks.append(("route_mutation", changed_route))
    changed_capability = copy.deepcopy(rows["A0"]); changed_capability[0]["capability"] = "tone_control"; attacks.append(("capability_mutation", changed_capability))
    for name, attack in attacks:
        try:
            _verify_rows(attack, probes, parent)
        except Phase3Error:
            hostile[name] = True
        else:
            hostile[name] = False
    passed = all(hostile.values()) and all(value["lower_95"] > 0 for value in paired.values()) and reconstructed["A0"]["functional_passes_v1"] == 1393 and reconstructed["A0"]["repetition_collapses_v2"] == 0 and reconstructed["A0"]["router_correct"] == 1400 and reconstructed["A0"]["strong_exact"] == 1000
    result = {"format": "abi-capability-compiler-phase3-route-isolated-verifier-result/1", "status": "PASS_INDEPENDENT_HOSTILE_RECONSTRUCTION" if passed else "FAIL_INDEPENDENT_HOSTILE_RECONSTRUCTION", "protocol_sha256": sha256_file(protocol_path), "reconstructed": reconstructed, "paired_A0_minus_controls": paired, "common_record_sequence_sha256": next(iter(sequences)), "hostile_mutations_rejected": hostile, "all_gates_passed": passed, "neural_training_performed": False, "historical_evidence_changed": False, "final_test_accessed": False}
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--protocol",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); root=Path.cwd().resolve(); print(json.dumps(run(root,root/args.protocol,root/args.output),indent=2,sort_keys=True))


if __name__ == "__main__": main()
