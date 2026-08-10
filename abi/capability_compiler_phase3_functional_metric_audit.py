"""Read-only V460/V461 functional surface-equivalence audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .capability_compiler_functional_v2 import evaluate_functional_v2
from .capability_compiler_phase2_common import evaluate_functional, sha256_file


SYSTEMS = {
    "teacher_T0": "results/abi_capability_compiler_phase2/teacher/T0/development_outputs.jsonl",
    "candidate_V459": "results/abi_capability_compiler_phase3_copy_balanced_transition/evaluation_v459/development_outputs.jsonl",
    "failed_A3": "results/abi_capability_compiler_phase3/evaluation_v4/A3-seed104729/development_outputs.jsonl",
    "failed_C3": "results/abi_capability_compiler_phase3_shared_output/evaluation_v11/C3-seed104729/development_outputs.jsonl"
}


def run(root: Path, protocol: Path, output: Path):
    catalog_path = root / "catalogs/capability_compiler_phase1_frozen_v1.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8")); probes = {row["probe_id"]: row for row in catalog["probes"] if row["split"] == "validation"}
    systems = {}
    for name, relative in SYSTEMS.items():
        path = root / relative; rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        v1 = Counter(); v2 = Counter()
        for row in rows:
            probe = probes[row["probe_id"]]; capability = row["capability"]
            v1[capability] += evaluate_functional(row["output"], probe["evaluator"])
            v2[capability] += evaluate_functional_v2(row["output"], probe["evaluator"], capability)
        systems[name] = {"path": relative, "sha256": sha256_file(path), "observations": len(rows), "v1_total": sum(v1.values()), "v2_total": sum(v2.values()), "v1_by_capability": dict(sorted(v1.items())), "v2_by_capability": dict(sorted(v2.items()))}
    gates = {"teacher_abstention_construct": systems["teacher_T0"]["v2_by_capability"]["abstention"] >= 95, "teacher_fluent_construct": systems["teacher_T0"]["v2_by_capability"]["fluent_realization"] >= 95, "v459_still_fails_abstention": systems["candidate_V459"]["v2_by_capability"]["abstention"] < 90, "v459_still_fails_coherence": systems["candidate_V459"]["v2_by_capability"]["coherence"] < 90, "failed_controls_remain_below_1000": systems["failed_A3"]["v2_total"] < 1000 and systems["failed_C3"]["v2_total"] < 1000}
    result = {"format": "abi-capability-compiler-functional-metric-audit/1", "status": "PASS_V2_SURFACE_EQUIVALENCE_CONSTRUCT_AUDIT" if all(gates.values()) else "FAIL_V2_SURFACE_EQUIVALENCE_AUDIT", "protocol_sha256": sha256_file(protocol), "systems": systems, "gates": gates, "audit_pass": all(gates.values()), "training_performed": False, "artifact_mutated": False, "final_test_accessed": False, "phase3_certified": False, "claim_boundary": "Exploratory symmetric development construct audit. Historical V1 scores remain unchanged and V459 remains nonpromotional."}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(".")); parser.add_argument("--protocol", type=Path, default=Path("ABI_CAPABILITY_COMPILER_PHASE3_FUNCTIONAL_METRIC_AUDIT_PROTOCOL_V460.json")); parser.add_argument("--output", type=Path, default=Path("results/abi_capability_compiler_phase3_functional_metric_audit/audit_v461/result.json")); args = parser.parse_args(); result = run(args.root, args.root / args.protocol, args.root / args.output); print(json.dumps({"status": result["status"], "gates": result["gates"], "systems": {k: {"v1": v["v1_total"], "v2": v["v2_total"]} for k, v in result["systems"].items()}}, indent=2, sort_keys=True))


if __name__ == "__main__": main()
