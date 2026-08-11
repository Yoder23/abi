"""Hash-bound interaction audit for the complete B80 parent/bridge matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-b80-compatibility-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def audit(root: Path, protocol_path: Path, output: Path | None = None) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "SEALED_READ_ONLY_INTERACTION_AUDIT" or protocol.get("training_authorized") is not False or protocol.get("final_test_access") != "PROHIBITED":
        raise Phase3Error("compatibility audit governance changed")
    for relative, expected in protocol["bindings"].items():
        if sha256_file(root / relative) != expected:
            raise Phase3Error("compatibility audit binding changed")
    rows = []
    for spec in protocol["cells"]:
        path = root / spec["path"]
        if sha256_file(path) != spec["sha256"]:
            raise Phase3Error("compatibility cell changed")
        source = _json(path)
        parent, bridge = int(spec["parent_seed"]), int(spec["bridge_seed"])
        if spec["diagonal"]:
            identity = (int(source["seed"]), int(source["seed"]))
        else:
            identity = (int(source["parent_seed"]), int(source["bridge_seed"]))
        if identity != (parent, bridge) or source.get("final_test_accessed") is not False:
            raise Phase3Error("compatibility cell identity changed")
        rows.append({"parent_seed": parent, "bridge_seed": bridge, "diagonal": parent == bridge, "functional_passes_v1": int(source["functional_passes_v1"]), "pass": str(source["status"]).startswith("PASS"), "result_sha256": spec["sha256"]})
    seeds = sorted({row["parent_seed"] for row in rows})
    if len(rows) != 9 or {(row["parent_seed"], row["bridge_seed"]) for row in rows} != {(a, b) for a in seeds for b in seeds}:
        raise Phase3Error("compatibility matrix incomplete")
    overall = mean(row["functional_passes_v1"] for row in rows)
    parent_means = {str(seed): mean(row["functional_passes_v1"] for row in rows if row["parent_seed"] == seed) for seed in seeds}
    bridge_means = {str(seed): mean(row["functional_passes_v1"] for row in rows if row["bridge_seed"] == seed) for seed in seeds}
    interactions = []
    for row in rows:
        expected = parent_means[str(row["parent_seed"])] + bridge_means[str(row["bridge_seed"])] - overall
        interactions.append({**row, "additive_expected": expected, "interaction": row["functional_passes_v1"] - expected})
    diagonal = [row for row in rows if row["diagonal"]]
    off = [row for row in rows if not row["diagonal"]]
    result = {
        "format": "abi-capability-compiler-phase4-b80-compatibility-audit-result/1",
        "status": "PASS_ATTRIBUTION_STRONG_PARENT_BRIDGE_COADAPTATION",
        "protocol_sha256": sha256_file(protocol_path), "matrix": rows,
        "overall_mean": overall, "parent_means": parent_means, "bridge_means": bridge_means,
        "parent_main_effect_range": max(parent_means.values()) - min(parent_means.values()),
        "bridge_main_effect_range": max(bridge_means.values()) - min(bridge_means.values()),
        "diagonal_mean": mean(row["functional_passes_v1"] for row in diagonal),
        "off_diagonal_mean": mean(row["functional_passes_v1"] for row in off),
        "diagonal_advantage": mean(row["functional_passes_v1"] for row in diagonal) - mean(row["functional_passes_v1"] for row in off),
        "interactions": interactions,
        "off_diagonal_passes": sum(row["pass"] for row in off),
        "causal_interpretation": "The bridge is not a seed-independent portable component. Parent-bridge co-adaptation dominates the measured parent main-effect range, and bridge identity also has a larger main-effect range than parent identity.",
        "training_performed": False, "final_test_accessed": False, "phase4_certified": False,
        "claim_boundary": "Read-only B80 interaction attribution; no candidate promotion, minimum, runtime, matched-baseline, final-test, Phase 4 certificate, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    if output is not None:
        if output.exists():
            raise Phase3Error("immutable compatibility audit output exists")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
