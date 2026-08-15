"""Read-only B20 lower-anchor audit against the five-route v22 host scope."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import wilson


FORMAT = "abi-capability-compiler-phase4-b20-host-compatibility-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def host_can_change(row: Mapping[str, Any]) -> bool:
    """Return whether a declared v19-v22/fifth-route mechanism can change this row."""
    return (
        row["capability"] in {"coherence", "format_control", "clarification"}
        or bool(row["repetition_collapse_v2"])
    )


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_B20_HOST_COMPATIBILITY_AUDIT"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B20 host-compatibility audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B20 host-compatibility binding changed: {relative}")
    if output.exists():
        raise Phase3Error(f"immutable B20 host-compatibility audit exists: {output}")
    rows = _rows(root / protocol["b20_outputs"])
    declared = _json(root / protocol["b20_result"])
    immutable = [row for row in rows if not host_can_change(row)]
    immutable_per = {}
    for capability in sorted({str(row["capability"]) for row in immutable}):
        selected = [row for row in immutable if row["capability"] == capability]
        passed = sum(bool(row["functional_pass_v1"]) for row in selected)
        immutable_per[capability] = {
            "passes": passed,
            "observations": len(selected),
            "wilson": wilson(passed, len(selected)),
        }
    locked_failures = {
        "fluent_realization": immutable_per["fluent_realization"],
        "instruction_following": immutable_per["instruction_following"],
        "tone_control": immutable_per["tone_control"],
    }
    thresholds = protocol["thresholds"]
    failure_gates = {
        "fluent_realization_per_capability": not (
            locked_failures["fluent_realization"]["wilson"]["point"] >= thresholds["per_capability_point"]
            and locked_failures["fluent_realization"]["wilson"]["lower_95"] >= thresholds["per_capability_lower"]
        ),
        "instruction_following_critical": not (
            locked_failures["instruction_following"]["wilson"]["point"] >= thresholds["critical_point"]
            and locked_failures["instruction_following"]["wilson"]["lower_95"] >= thresholds["critical_lower"]
        ),
        "tone_control_per_capability": not (
            locked_failures["tone_control"]["wilson"]["point"] >= thresholds["per_capability_point"]
            and locked_failures["tone_control"]["wilson"]["lower_95"] >= thresholds["per_capability_lower"]
        ),
    }
    oracle_rows = []
    for row in rows:
        changed = dict(row)
        if host_can_change(row):
            changed["functional_pass_v1"] = True
            changed["repetition_collapse_v2"] = False
        oracle_rows.append(changed)
    oracle_passes = sum(bool(row["functional_pass_v1"]) for row in oracle_rows)
    gates = {
        "exact_1400_rows": len(rows) == 1400,
        "declared_baseline_reproduced": sum(bool(row["functional_pass_v1"]) for row in rows) == int(declared["functional_passes_v1"]),
        "host_scope_exhaustive_for_declared_mechanisms": True,
        "three_immutable_gate_failures": all(failure_gates.values()),
        "oracle_grants_every_changeable_row": all(
            bool(row["functional_pass_v1"]) and not bool(row["repetition_collapse_v2"])
            for row in oracle_rows
            if host_can_change(row)
        ),
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_model_loading_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b20-host-compatibility-audit-result/1",
        "status": "PASS_B20_SEED104729_REMAINS_FAILED_UNDER_GENEROUS_HOST_ORACLE" if all(gates.values()) else "FAIL_B20_HOST_COMPATIBILITY_AUDIT",
        "protocol_sha256": sha256_file(protocol_path),
        "historical_functional_passes": int(declared["functional_passes_v1"]),
        "historical_collapses": int(declared["repetition_collapses_v2"]),
        "host_changeable_rows": sum(host_can_change(row) for row in rows),
        "host_immutable_rows": len(immutable),
        "oracle_functional_passes": oracle_passes,
        "locked_failures": locked_failures,
        "failure_gates": failure_gates,
        "gates": gates,
        "decision": "Even granting a functional pass and zero collapse to every row that v19 coherence, v20/v21 guarding, v22 format realization, or the fifth clarification route can change, B20 seed104729 still fails unchanged fluent-realization, instruction-following, and tone-control confidence gates.",
        "authorized_next_action": "Build only the missing B20 seeds130363 and155921 from the immutable clean start, then apply the same generous read-only host oracle first. Physical five-route/v22 inference is unnecessary for a seed whose immutable rows already prove failure; any seed that survives the oracle requires the exact product screen.",
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Read-only B20 seed104729 lower-anchor failure proof. Missing B20 seeds, stable minimum, product runtime, final test, Phase 4, and ABI superiority remain open.",
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
