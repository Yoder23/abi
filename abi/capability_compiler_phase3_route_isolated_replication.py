"""Three-seed replication reducer for the Phase 3 route-isolated endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .capability_compiler_phase2_common import canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_route_isolated import CONTROL_SYSTEMS, FORMAT as ROUTE_FORMAT
from . import capability_compiler_phase3_final_controls as controls
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import paired_stratified_bootstrap


FORMAT = "abi-capability-compiler-phase3-route-isolated-replication-protocol/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["probe_id"]): row for row in map(json.loads, path.open(encoding="utf-8"))}
    if len(rows) != 1400:
        raise Phase3Error(f"replication depth changed: {path}")
    return rows


def _verify_embedded_hash(document: Mapping[str, Any]) -> None:
    payload = dict(document)
    expected = payload.pop("evidence_sha256", None)
    actual = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    if expected != actual:
        raise Phase3Error("embedded evidence hash changed")


def hierarchical_bootstrap(
    seed_differences: list[dict[str, list[int]]], *, replicates: int, seed: int
) -> dict[str, Any]:
    if len(seed_differences) < 2 or replicates < 1000:
        raise Phase3Error("hierarchical bootstrap depth changed")
    capabilities = tuple(sorted(seed_differences[0]))
    if any(tuple(sorted(item)) != capabilities for item in seed_differences):
        raise Phase3Error("capability strata changed")
    generator = np.random.default_rng(seed)
    seed_bootstraps = np.empty((len(seed_differences), replicates), dtype=np.float64)
    all_values: list[int] = []
    for seed_index, by_capability in enumerate(seed_differences):
        capability_means = []
        for capability in capabilities:
            values = np.asarray(by_capability[capability], dtype=np.float64)
            if values.size != 100:
                raise Phase3Error("prompt depth per capability changed")
            all_values.extend(int(value) for value in values)
            indices = generator.integers(0, values.size, size=(replicates, values.size))
            capability_means.append(values[indices].mean(axis=1))
        seed_bootstraps[seed_index] = np.stack(capability_means).mean(axis=0)
    selected_seeds = generator.integers(0, len(seed_differences), size=(replicates, len(seed_differences)))
    columns = np.arange(replicates)[:, None]
    distribution = seed_bootstraps[selected_seeds, columns].mean(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {
        "method": "training-seed-and-capability-stratified-prompt-paired-percentile-bootstrap",
        "replicates": replicates,
        "seed": seed,
        "training_seed_count": len(seed_differences),
        "prompt_observations": len(all_values),
        "candidate_minus_control": float(np.mean(all_values)),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_THREE_PAIRED_SEED_REPLICATION"
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or len(protocol.get("replications", ())) != 3
    ):
        raise Phase3Error("three-seed replication governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"replication binding changed: {relative}")
    return protocol, sha256_file(path)


def decide(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256 = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable replication decision exists")
    seed_summaries: list[dict[str, Any]] = []
    differences: dict[str, list[dict[str, list[int]]]] = {system: [] for system in CONTROL_SYSTEMS}
    seen_seeds: set[int] = set()

    for replication in protocol["replications"]:
        training_seed = int(replication["training_seed"])
        if training_seed in seen_seeds:
            raise Phase3Error("training seed repeated")
        seen_seeds.add(training_seed)
        decision_path = root / replication["decision"]
        decision = _json(decision_path)
        _verify_embedded_hash(decision)
        if sha256_file(decision_path) != replication["decision_sha256"] or decision.get("evidence_sha256") != replication["evidence_sha256"]:
            raise Phase3Error("paired-seed decision binding changed")
        route_protocol_path = root / replication["route_protocol"]
        route_protocol = _json(route_protocol_path)
        route_protocol_sha256 = sha256_file(route_protocol_path)
        if (
            route_protocol.get("format") != ROUTE_FORMAT
            or route_protocol.get("final_test_access") != "PROHIBITED"
            or route_protocol.get("status") not in {
                "PREREGISTERED_ROUTE_ISOLATED_MATCHED_CONTROLS",
                "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX",
            }
        ):
            raise Phase3Error("historical route protocol governance changed")
        _, _, base = controls.load_protocol(root, root / route_protocol["base_control_protocol"])
        if route_protocol_sha256 != decision["protocol_sha256"] or not decision["all_controls_passed"]:
            raise Phase3Error("paired-seed decision did not pass or protocol changed")

        a0_path = root / route_protocol["A0_outputs"]
        a0_rows = _rows(a0_path)
        a0_evaluation = _json(a0_path.parent / "result.json")
        if a0_evaluation["raw_outputs_sha256"] != sha256_file(a0_path) or a0_evaluation["checkpoint_sha256"] != decision["A0"]["checkpoint_sha256"]:
            raise Phase3Error("A0 replication binding changed")
        thresholds = base["absolute_screen"]
        absolute_gates = {
            "per_capability": all(
                value["wilson_v1"]["point"] >= thresholds["per_capability_functional_point_estimate_minimum"]
                and value["wilson_v1"]["lower_95"] >= thresholds["per_capability_functional_wilson_lower_minimum"]
                for value in a0_evaluation["per_capability"].values()
            ),
            "critical": all(
                a0_evaluation["per_capability"][name]["wilson_v1"]["point"] >= thresholds["critical_point_minimum"]
                and a0_evaluation["per_capability"][name]["wilson_v1"]["lower_95"] >= thresholds["critical_wilson_lower_minimum"]
                for name in ("prompt_grounding", "instruction_following", "abstention")
            ),
            "zero_collapse": a0_evaluation["repetition_collapses_v2"] == 0,
            "router_exact": a0_evaluation["router_correct"] == 1400,
            "strong_exact": a0_evaluation["strong_routes_exact"] == 1000,
        }
        probes = {str(row["probe_id"]): row for row in development_probes(root / base["development"]["catalog_path"])}
        teacher = {str(row["probe_id"]): row for row in map(json.loads, (root / base["development"]["teacher_reference"]).open(encoding="utf-8"))}
        teacher_paired = [
            {
                "capability": row["capability"],
                "candidate_pass": bool(row["functional_pass_v1"]),
                "teacher_pass": evaluate_functional(str(teacher[probe_id]["output"]), probes[probe_id]["evaluator"]),
            }
            for probe_id, row in a0_rows.items()
        ]
        teacher_comparison = paired_stratified_bootstrap(
            teacher_paired, replicates=10_000, seed=5350000 + training_seed
        )
        teacher_gate = teacher_comparison["lower_95"] >= base["teacher_relative_screen"]["candidate_minus_teacher_lower_95_minimum"]

        for system in CONTROL_SYSTEMS:
            directory = root / route_protocol["control_outputs"][system]
            control_rows = _rows(directory / "development_outputs.jsonl")
            by_capability = {name: [] for name in sorted({str(row["capability"]) for row in a0_rows.values()})}
            for probe_id, a0_row in a0_rows.items():
                control_row = control_rows[probe_id]
                if a0_row["capability"] != control_row["capability"]:
                    raise Phase3Error("replication capability join changed")
                by_capability[str(a0_row["capability"])].append(
                    int(bool(a0_row["functional_pass_v1"])) - int(bool(control_row["functional_pass_v1"]))
                )
            differences[system].append(by_capability)

        seed_summaries.append(
            {
                "training_seed": training_seed,
                "A0_functional_passes_v1": a0_evaluation["functional_passes_v1"],
                "A0_checkpoint_sha256": a0_evaluation["checkpoint_sha256"],
                "A0_outputs_sha256": a0_evaluation["raw_outputs_sha256"],
                "paired_controls_passed": True,
                "absolute_gates": absolute_gates,
                "teacher_comparison": teacher_comparison,
                "teacher_noninferior": teacher_gate,
                "seed_passed": all(absolute_gates.values()) and teacher_gate,
            }
        )

    aggregate = {
        system: hierarchical_bootstrap(values, replicates=10_000, seed=5351729 + index)
        for index, (system, values) in enumerate(differences.items())
    }
    aggregate_gates = {system: comparison["lower_95"] > 0.0 for system, comparison in aggregate.items()}
    gates = {
        "three_distinct_training_seeds": len(seen_seeds) == 3,
        "all_seed_absolute_and_teacher_gates": all(seed["seed_passed"] for seed in seed_summaries),
        "all_seed_paired_control_gates": all(seed["paired_controls_passed"] for seed in seed_summaries),
        "all_hierarchical_control_gates": all(aggregate_gates.values()),
        "final_test_not_accessed": True,
    }
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase3-route-isolated-replication-decision/1",
        "status": "PASS_THREE_PAIRED_SEED_ROUTE_ISOLATED_REPLICATION" if passed else "FAIL_THREE_PAIRED_SEED_ROUTE_ISOLATED_REPLICATION",
        "protocol_sha256": protocol_sha256,
        "seeds": seed_summaries,
        "hierarchical_A0_minus_control": aggregate,
        "hierarchical_gates": aggregate_gates,
        "gates": gates,
        "replication_passed": passed,
        "final_test_accessed": False,
        "historical_evidence_changed": False,
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
    result = decide(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
