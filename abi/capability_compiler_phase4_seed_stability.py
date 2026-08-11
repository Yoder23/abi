"""Hash-bound read-only attribution for the Phase 4 ABI frontier instability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT = "abi-capability-compiler-phase4-seed-stability/1"


class SeedStabilityError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SeedStabilityError(f"unsafe path: {relative}")
    if not path.is_file():
        raise SeedStabilityError(f"missing evidence: {relative}")
    return path


def load_protocol(root: Path, protocol_path: Path) -> Mapping[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("format") != FORMAT or protocol.get("status") != "SEALED_READ_ONLY_ATTRIBUTION":
        raise SeedStabilityError("seed-stability governance changed")
    for relative, expected in protocol["bindings"].items():
        if sha256_file(_safe(root, relative)) != expected:
            raise SeedStabilityError(f"binding changed: {relative}")
    return protocol


def _load_bound(root: Path, row: Mapping[str, Any], key: str) -> tuple[Path, Mapping[str, Any]]:
    spec = row[key]
    path = _safe(root, str(spec["path"]))
    if sha256_file(path) != spec["sha256"]:
        raise SeedStabilityError(f"run evidence changed: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def _stage(eval_result: Mapping[str, Any], *, final: bool) -> dict[str, Any]:
    passes_key = "passes_v1" if final else "passes"
    total_key = "functional_passes_v1" if final else "functional_passes"
    collapse_key = "collapses_v2" if final else "v2_collapses"
    per = {
        capability: {
            "passes": int(values[passes_key]),
            "collapses": int(values[collapse_key]),
        }
        for capability, values in eval_result["per_capability"].items()
    }
    return {
        "functional_passes": int(eval_result[total_key]),
        "repetition_collapses_v2": int(eval_result["repetition_collapses_v2"]),
        "per_capability": per,
    }


def analyze(root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    runs = []
    for spec in protocol["runs"]:
        _, result = _load_bound(root, spec, "result")
        _, intermediate_eval = _load_bound(root, spec, "intermediate_evaluation")
        _, final_eval = _load_bound(root, spec, "final_evaluation")
        budget = str(spec["budget"])
        seed = int(spec["seed"])
        if result["budget"]["id"] != budget or int(result["seed"]) != seed:
            raise SeedStabilityError("budget or seed identity changed")
        if result["stage_checkpoints"]["v463"] != intermediate_eval["checkpoint_sha256"]:
            raise SeedStabilityError("intermediate checkpoint lineage changed")
        if result["stage_checkpoints"]["v526"] != final_eval["checkpoint_sha256"]:
            raise SeedStabilityError("final checkpoint lineage changed")
        if result["final_test_accessed"] or intermediate_eval["final_test_accessed"] or final_eval["final_test_accessed"]:
            raise SeedStabilityError("final-test firewall violated")
        gate_pass = all(bool(value) for value in result["gates"].values())
        status_pass = result["status"] == "PASS_PHASE4_ABI_BUDGET_MACHINE_GATES"
        if gate_pass != status_pass:
            raise SeedStabilityError("result status disagrees with gates")
        runs.append({
            "budget": budget,
            "seed": seed,
            "passed": status_pass,
            "intermediate": _stage(intermediate_eval, final=False),
            "final": _stage(final_eval, final=True),
            "teacher_lower_95": float(result["teacher_comparison_v1"]["lower_95"]),
            "wall_seconds": float(result["wall_seconds"]),
            "checkpoint_sha256": result["stage_checkpoints"]["v526"],
        })
    expected = {(str(row["budget"]), int(row["seed"])) for row in protocol["runs"]}
    actual = {(row["budget"], row["seed"]) for row in runs}
    if expected != actual or len(actual) != len(runs):
        raise SeedStabilityError("run matrix changed")

    budgets: dict[str, Any] = {}
    delta_candidates = []
    for budget in sorted({row["budget"] for row in runs}):
        subset = sorted((row for row in runs if row["budget"] == budget), key=lambda row: row["seed"])
        capabilities = sorted(subset[0]["final"]["per_capability"])
        final_ranges = {}
        delta_ranges = {}
        for capability in capabilities:
            finals = [row["final"]["per_capability"][capability]["passes"] for row in subset]
            deltas = [
                row["final"]["per_capability"][capability]["passes"]
                - row["intermediate"]["per_capability"][capability]["passes"]
                for row in subset
            ]
            final_ranges[capability] = max(finals) - min(finals)
            delta_ranges[capability] = {
                "by_seed": {str(row["seed"]): delta for row, delta in zip(subset, deltas)},
                "range": max(deltas) - min(deltas),
            }
            delta_candidates.append((max(deltas) - min(deltas), budget, capability, deltas))
        totals = [row["final"]["functional_passes"] for row in subset]
        budgets[budget] = {
            "seed_status": {str(row["seed"]): "PASS" if row["passed"] else "FAIL" for row in subset},
            "pass_count": sum(row["passed"] for row in subset),
            "all_seed_pass": all(row["passed"] for row in subset),
            "all_seed_fail": not any(row["passed"] for row in subset),
            "final_functional_passes": {str(row["seed"]): row["final"]["functional_passes"] for row in subset},
            "final_functional_range": max(totals) - min(totals),
            "final_collapses": {str(row["seed"]): row["final"]["repetition_collapses_v2"] for row in subset},
            "most_variable_final_capability": max(final_ranges, key=lambda name: (final_ranges[name], name)),
            "final_capability_ranges": final_ranges,
            "bridge_delta_ranges": delta_ranges,
        }
    most_unstable = max(delta_candidates, key=lambda item: (item[0], item[1], item[2]))
    boundary = protocol["tested_boundary"]
    reproduced = budgets[boundary["passing_budget"]]["all_seed_pass"] and budgets[boundary["adjacent_lower"]]["all_seed_fail"]
    result = {
        "format": "abi-capability-compiler-phase4-seed-stability-result/1",
        "status": "PASS_REPRODUCED_ABI_FRONTIER" if reproduced else "FAIL_NO_REPRODUCED_ABI_FRONTIER",
        "protocol_sha256": sha256_file(_safe(root, str(protocol["self_path"]))),
        "runs": runs,
        "budgets": budgets,
        "tested_boundary": boundary,
        "frontier_reproduced": reproduced,
        "most_seed_sensitive_bridge_effect": {
            "budget": most_unstable[1],
            "capability": most_unstable[2],
            "delta_range": most_unstable[0],
            "deltas_in_seed_order": most_unstable[3],
        },
        "measured_bottleneck": "The final recovery bridge has a seed-dependent capability effect; it improves aggregate quality but does not preserve the same capability floors across seeds.",
        "final_test_accessed": False,
        "training_performed": False,
        "phase4_certified": False,
        "next_action": "Preregister at most one final-bridge stabilization method targeted at the measured seed-dependent bridge effect; do not expand information budgets, tune thresholds, run matched baselines, or access final data first.",
        "claim_boundary": "Read-only six-run ABI-arm seed-stability attribution; no stable frontier, minimum, runtime, matched-baseline comparison, Phase 4 certificate, final-test result, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def run(root: Path, protocol_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = load_protocol(root, protocol_path)
    result = analyze(root, protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    if path.exists():
        raise SeedStabilityError(f"immutable result exists: {path}")
    path.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, Path(args.protocol).resolve(), Path(args.output_dir).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS_REPRODUCED_ABI_FRONTIER" else 1


if __name__ == "__main__":
    raise SystemExit(main())
