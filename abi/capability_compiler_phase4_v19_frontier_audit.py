"""Read-only eligibility audit for applying v19 to frozen B40/B80 seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-v19-frontier-audit/1"
BRACKETED = re.compile(r"\[[^\[\]\r\n]{1,128}\]")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def eligible(prompt: str) -> bool:
    spans = BRACKETED.findall(prompt)
    return "return the labels in order" in prompt.casefold() and len(spans) == len(set(spans)) == 3


def run(root: Path, protocol_path: Path, output: Path):
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_READ_ONLY_V19_FRONTIER_ELIGIBILITY" or protocol.get("model_inference_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("v19 frontier audit governance changed")
    for relative, expected in protocol["bindings"].items():
        if sha256_file((root / relative).resolve()) != expected:
            raise Phase3Error(f"v19 frontier audit binding changed: {relative}")
    if output.exists():
        raise Phase3Error(f"immutable v19 frontier audit exists: {output}")
    catalog = _json(root / protocol["catalog"])
    probes = {row["probe_id"]: row for row in catalog["probes"] if row["split"] == "validation"}
    systems = []
    for spec in protocol["systems"]:
        rows = _rows(root / spec["outputs"])
        coherence = [row for row in rows if row["capability"] == "coherence"]
        eligible_rows = [row for row in coherence if eligible(str(probes[row["probe_id"]]["prompt"]))]
        failures = [row for row in rows if not row["functional_pass_v1"]]
        systems.append({
            "budget": spec["budget"],
            "seed": spec["seed"],
            "lineage_dir": spec["lineage_dir"],
            "outputs_sha256": protocol["bindings"][spec["outputs"]],
            "functional_passes": sum(row["functional_pass_v1"] for row in rows),
            "repetition_collapses": sum(row["repetition_collapse_v2"] for row in rows),
            "coherence_passes": sum(row["functional_pass_v1"] for row in coherence),
            "v19_eligible_coherence_rows": len(eligible_rows),
            "coherence_failures_requiring_inference": sum(not row["functional_pass_v1"] for row in eligible_rows),
            "noncoherence_failures_unchanged_by_v19": sum(row["capability"] != "coherence" for row in failures),
            "hypothetical_upper_passes_if_all_v19_rows_pass": sum(row["functional_pass_v1"] for row in rows) + sum(not row["functional_pass_v1"] for row in eligible_rows),
        })
    gates = {
        "six_registered_systems": len(systems) == 6,
        "all_coherence_rows_eligible": all(row["v19_eligible_coherence_rows"] == 100 for row in systems),
        "b80_seed130363_failure_in_scope": next(row for row in systems if row["budget"] == "B80" and row["seed"] == 130363)["coherence_failures_requiring_inference"] == 22,
        "noncoherence_evidence_immutable": True,
        "model_inference_absent": True,
        "training_absent": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-v19-frontier-audit-result/1",
        "status": "PASS_B40_B80_V19_RESREEN_REQUIRED" if all(gates.values()) else "FAIL_FRONTIER_AUDIT",
        "protocol_sha256": sha256_file(protocol_path),
        "systems": systems,
        "gates": gates,
        "authorized_rescreen_rows": 600,
        "model_inference_performed": False,
        "training_performed": False,
        "final_test_accessed": False,
        "claim_boundary": "Read-only v19 predicate eligibility only; hypothetical upper scores are not measured quality and no frontier or Phase 4 claim is made.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None):
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
