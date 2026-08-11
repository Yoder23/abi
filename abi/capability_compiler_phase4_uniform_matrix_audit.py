"""Fail-closed audit of the six-run Phase 4 uniform-exposure matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-uniform-matrix-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "SEALED_READ_ONLY_SIX_RUN_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("uniform matrix audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"uniform matrix audit binding changed: {relative}")
    return protocol, sha256_file(path)


def audit(root: Path, protocol_path: Path, output: Path | None = None) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    rows = []
    for spec in protocol["runs"]:
        path = root / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise Phase3Error("registered result changed")
        result = _json(path)
        expected_identity = (str(spec["budget"]), int(spec["seed"]))
        identity = (str(result.get("budget", {}).get("id")), int(result.get("seed", -1)))
        if identity != expected_identity or result.get("protocol_sha256") != protocol["stabilization_protocol_sha256"]:
            raise Phase3Error("registered result lineage changed")
        if result.get("final_test_accessed") is not False or result.get("new_teacher_information") != 0:
            raise Phase3Error("registered result crossed information firewall")
        if result.get("exposure_balance", {}).get("maximum_within_stratum_exposure_range") != 1:
            raise Phase3Error("registered result lost exposure invariant")
        all_gates = all(bool(value) for value in result["gates"].values())
        if all_gates != str(result["status"]).startswith("PASS"):
            raise Phase3Error("registered result status disagrees with gates")
        rows.append({
            "budget": expected_identity[0], "seed": expected_identity[1], "pass": all_gates,
            "functional_passes_v1": int(result["functional_passes_v1"]),
            "repetition_collapses_v2": int(result["repetition_collapses_v2"]),
            "teacher_relative_lower_95": float(result["teacher_comparison_v1"]["lower_95"]),
            "result_sha256": spec["sha256"],
        })
    matrices = {
        budget: ["PASS" if row["pass"] else "FAIL" for row in sorted(rows, key=lambda item: item["seed"]) if row["budget"] == budget]
        for budget in ("B40", "B80")
    }
    success = matrices["B80"] == ["PASS"] * 3 and matrices["B40"] == ["FAIL"] * 3
    result = {
        "format": "abi-capability-compiler-phase4-uniform-matrix-audit-result/1",
        "status": "PASS_STABLE_B80_FRONTIER" if success else "FAIL_STABILIZATION_REJECTED_NO_STABLE_FRONTIER",
        "protocol_sha256": protocol_sha,
        "runs": rows,
        "matrix": matrices,
        "success_rule_satisfied": success,
        "historical_matrix": protocol["historical_matrix"],
        "matrix_changed_from_historical": matrices != protocol["historical_matrix"],
        "all_source_hashes_verified": True,
        "all_exposure_ranges_at_most_one": True,
        "new_teacher_information": 0,
        "training_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "decision": "REJECT_UNIFORM_EXPOSURE_STABILIZATION_CLOSE_BRANCH_NO_SWEEP" if not success else "ADVANCE_STABLE_ABI_ARM_TO_RUNTIME_AND_MATCHED_BASELINES",
        "claim_boundary": "Read-only six-run ABI-arm audit; no minimum, runtime, matched-baseline, final-test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if output is not None:
        if output.exists():
            raise Phase3Error("immutable uniform matrix audit output exists")
        output.mkdir(parents=True)
        _write_immutable(output / "result.json", json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = audit(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
