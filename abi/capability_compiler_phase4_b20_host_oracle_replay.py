"""Replay the generous declared-host oracle on one clean-start B20 seed."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson
from .capability_compiler_phase4_b20_host_compatibility_audit import host_can_change


FORMAT = "abi-capability-compiler-phase4-b20-host-oracle-replay/1"
CRITICAL = {"prompt_grounding", "instruction_following", "abstention"}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def capability_metrics(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for capability in sorted({str(row["capability"]) for row in rows}):
        selected = [row for row in rows if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in selected)
        result[capability] = {
            "passes": passed,
            "observations": len(selected),
            "wilson": wilson(passed, len(selected)),
        }
    return result


def oracle_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    replayed: list[dict[str, Any]] = []
    for row in rows:
        changed = dict(row)
        if host_can_change(row):
            changed["functional_pass_v1"] = True
            changed["functional_pass_v2"] = True
            changed["repetition_collapse_v2"] = False
        replayed.append(changed)
    return replayed


def _gate_failures(metrics: Mapping[str, Mapping[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    failures: dict[str, Any] = {}
    for capability, value in metrics.items():
        estimate = value["wilson"]
        per_capability_pass = (
            float(estimate["point"]) >= float(thresholds["per_capability_point"])
            and float(estimate["lower_95"]) >= float(thresholds["per_capability_lower"])
        )
        critical_pass = True
        if capability in CRITICAL:
            critical_pass = (
                float(estimate["point"]) >= float(thresholds["critical_point"])
                and float(estimate["lower_95"]) >= float(thresholds["critical_lower"])
            )
        if not per_capability_pass or not critical_pass:
            failures[capability] = {
                **value,
                "per_capability_pass": per_capability_pass,
                "critical_pass": critical_pass,
            }
    return failures


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_B20_HOST_ORACLE_REPLAY"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B20 host-oracle replay governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B20 host-oracle replay binding changed: {relative}")
    if output.exists():
        raise Phase3Error(f"immutable B20 host-oracle replay exists: {output}")

    rows_path = root / protocol["b20_outputs"]
    result_path = root / protocol["b20_result"]
    rows = _rows(rows_path)
    declared = _json(result_path)
    replayed = oracle_rows(rows)
    original_metrics = capability_metrics(rows)
    oracle_metrics = capability_metrics(replayed)
    failures = _gate_failures(oracle_metrics, protocol["thresholds"])
    expected_capabilities = set(protocol["expected_capabilities"])

    original_passes = sum(bool(row["functional_pass_v1"]) for row in rows)
    original_collapses = sum(bool(row["repetition_collapse_v2"]) for row in rows)
    replayed_passes = sum(bool(row["functional_pass_v1"]) for row in replayed)
    replayed_collapses = sum(bool(row["repetition_collapse_v2"]) for row in replayed)
    gates = {
        "exact_1400_distinct_rows": len(rows) == 1400 and len({str(row["probe_id"]) for row in rows}) == 1400,
        "exact_capability_matrix": set(original_metrics) == expected_capabilities and all(value["observations"] == 100 for value in original_metrics.values()),
        "declared_functional_total_reproduced": original_passes == int(declared["functional_passes_v1"]),
        "declared_collapse_total_reproduced": original_collapses == int(declared["repetition_collapses_v2"]),
        "declared_checkpoint_reproduced": str(rows[0]["checkpoint_sha256"] if "checkpoint_sha256" in rows[0] else declared["checkpoint_sha256"]) == str(declared["checkpoint_sha256"]),
        "host_scope_is_unchanged": all(host_can_change(row) == (row["capability"] in {"coherence", "format_control", "clarification"} or bool(row["repetition_collapse_v2"])) for row in rows),
        "oracle_grants_every_changeable_row": all(bool(row["functional_pass_v1"]) and bool(row["functional_pass_v2"]) and not bool(row["repetition_collapse_v2"]) for row in replayed if host_can_change(row)),
        "oracle_changes_no_immutable_row": all(before == after for before, after in zip(rows, replayed) if not host_can_change(before)),
        "oracle_zero_collapse": replayed_collapses == 0,
        "at_least_one_immutable_quality_gate_still_fails": bool(failures),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    seed = int(protocol["seed"])
    passed = all(gates.values())
    result = {
        "format": "abi-capability-compiler-phase4-b20-host-oracle-replay-result/1",
        "status": f"PASS_B20_SEED{seed}_REMAINS_FAILED_UNDER_GENEROUS_HOST_ORACLE" if passed else "FAIL_B20_HOST_ORACLE_REPLAY",
        "protocol_sha256": sha256_file(protocol_path),
        "seed": seed,
        "inputs": {
            "outputs": protocol["b20_outputs"],
            "outputs_sha256": sha256_file(rows_path),
            "result": protocol["b20_result"],
            "result_sha256": sha256_file(result_path),
        },
        "historical_functional_passes": original_passes,
        "historical_collapses": original_collapses,
        "host_changeable_rows": sum(host_can_change(row) for row in rows),
        "host_immutable_rows": sum(not host_can_change(row) for row in rows),
        "oracle_functional_passes": replayed_passes,
        "oracle_collapses": replayed_collapses,
        "original_per_capability": original_metrics,
        "oracle_per_capability": oracle_metrics,
        "oracle_gate_failures": failures,
        "gates": gates,
        "decision": "This B20 seed remains below the locked absolute quality boundary even after granting perfect functional and collapse outcomes to every row changeable by the complete declared host mechanisms." if passed else "The read-only oracle does not prove this B20 seed insufficient; exact five-route product inference is required.",
        "authorized_next_action": protocol["on_pass_authorized_next_action"] if passed else protocol["on_fail_authorized_next_action"],
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
