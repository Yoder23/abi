"""Test the frozen collapse guard on every capability-bank output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import CAPABILITIES, canonical_json_bytes, evaluate_functional, sha256_file
from .capability_compiler_phase2_teacher import development_probes
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_contract_guard_v2_audit import truncate_at_first_v2_collapse
from .capability_compiler_phase3_final_controls import wilson
from .capability_compiler_phase3_repetition_metric_audit import repetition_collapse_v2


FORMAT = "abi-capability-compiler-phase4-universal-guard-feasibility/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_UNIVERSAL_GUARD" or protocol.get("training_authorized") is not False or protocol.get("promotion_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("universal-guard feasibility governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"universal-guard feasibility binding changed: {relative}")
    return protocol, sha256_file(path)


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable universal-guard output exists")
    probes = {row["probe_id"]: row for row in development_probes(root / protocol["catalog"])}
    source = [json.loads(line) for line in (root / protocol["selected_outputs"]).read_text(encoding="utf-8").splitlines() if line]
    rows = []
    for row in source:
        value, terminated = truncate_at_first_v2_collapse(str(row["output"]))
        rows.append({**row, "output": value, "guard_terminated": terminated, "functional_pass_v1": evaluate_functional(value, probes[row["probe_id"]]["evaluator"]), "repetition_collapse_v2": repetition_collapse_v2(value)})
    per = {}
    for capability in CAPABILITIES:
        values = [row for row in rows if row["capability"] == capability]
        passed = sum(row["functional_pass_v1"] for row in values)
        per[capability] = {"passes": passed, "observations": len(values), "wilson": wilson(passed, len(values)), "collapses": sum(row["repetition_collapse_v2"] for row in values)}
    t = protocol["thresholds"]
    gates = {
        "per_capability": all(value["wilson"]["point"] >= t["per_capability_point"] and value["wilson"]["lower_95"] >= t["per_capability_lower"] for value in per.values()),
        "critical": all(per[name]["wilson"]["point"] >= t["critical_point"] and per[name]["wilson"]["lower_95"] >= t["critical_lower"] for name in protocol["critical_capabilities"]),
        "zero_collapse": sum(row["repetition_collapse_v2"] for row in rows) == 0,
        "functional_quality_not_reduced": sum(row["functional_pass_v1"] for row in rows) >= int(protocol["source_functional_passes"]),
        "training_not_performed": True,
        "promotion_not_authorized": True,
        "final_test_not_accessed": True,
    }
    raw = output.parent / "guarded_outputs.jsonl"
    output.parent.mkdir(parents=True)
    _write_immutable(raw, b"".join(canonical_json_bytes(row) for row in rows))
    result = {
        "format": "abi-capability-compiler-phase4-universal-guard-feasibility-result/1",
        "status": "PASS_CAPABILITY_ISOLATION_PLUS_UNIVERSAL_GUARD_DESIGN_SUPPORTED" if all(gates.values()) else "FAIL_UNIVERSAL_GUARD_NOT_SUPPORTED",
        "protocol_sha256": protocol_sha,
        "functional_passes": sum(row["functional_pass_v1"] for row in rows),
        "observations": len(rows),
        "guard_terminations": sum(row["guard_terminated"] for row in rows),
        "repetition_collapses_v2": sum(row["repetition_collapse_v2"] for row in rows),
        "per_capability": per,
        "gates": gates,
        "raw_outputs_sha256": sha256_file(raw),
        "training_performed": False,
        "promotion_authorized": False,
        "final_test_accessed": False,
        "interpretation": "A pass supports one future sparse capability-isolated trainable artifact with the frozen guard applied capability-independently. It does not validate or promote this post-hoc bank.",
        "claim_boundary": "Read-only feasibility only; no trained candidate, stable frontier, minimum, matched baseline, final test, Phase 4 certificate, or superiority claim."
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
