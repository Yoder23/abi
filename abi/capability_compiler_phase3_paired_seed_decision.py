"""Reduce a frozen route-isolated paired-seed matrix without changing its protocol.

The original V532 replication protocols deliberately bind A0 through separate
output and metadata fields.  This reducer preserves those immutable protocols
and performs the same locked prompt-paired causal comparisons as V529.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase3_route_isolated import (
    CONTROL_SYSTEMS,
    FORMAT,
    SYSTEM,
    load_protocol,
)
from .capability_compiler_phase3_routed_v15_autonomous_screen_isolated import (
    paired_stratified_bootstrap,
)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = {str(row["probe_id"]): row for row in map(json.loads, path.open(encoding="utf-8"))}
    if len(rows) != 1400:
        raise Phase3Error(f"paired output depth changed: {path}")
    return rows


def decide(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha256, _ = load_protocol(root, protocol_path)
    if protocol.get("format") != FORMAT or protocol.get("status") != "PREREGISTERED_ROUTE_ISOLATED_PAIRED_SEED_MATRIX":
        raise Phase3Error("paired-seed decision is not authorized")
    if output.exists():
        raise Phase3Error("immutable paired-seed decision exists")

    a0_path = root / protocol["A0_outputs"]
    a0 = _rows(a0_path)
    a0_evaluation = _json(a0_path.parent / "result.json")
    a0_metadata = _json(root / protocol["A0_metadata"])
    a0_checkpoint = str(a0_metadata["checkpoint"]["sha256"])
    if (
        a0_evaluation.get("system") != SYSTEM
        or a0_evaluation.get("protocol_sha256") != protocol_sha256
        or a0_evaluation.get("checkpoint_sha256") != a0_checkpoint
        or a0_evaluation.get("raw_outputs_sha256") != sha256_file(a0_path)
    ):
        raise Phase3Error("A0 evidence binding changed")

    comparisons: dict[str, Any] = {}
    systems: dict[str, Any] = {}
    for index, system in enumerate(CONTROL_SYSTEMS):
        directory = root / protocol["control_outputs"][system]
        result = _json(directory / "result.json")
        output_path = directory / "development_outputs.jsonl"
        rows = _rows(output_path)
        if (
            set(rows) != set(a0)
            or result.get("system") != system
            or result.get("protocol_sha256") != protocol_sha256
            or result.get("raw_outputs_sha256") != sha256_file(output_path)
        ):
            raise Phase3Error(f"{system} evidence binding changed")
        paired = []
        for probe_id, candidate in a0.items():
            control = rows[probe_id]
            if candidate["capability"] != control["capability"]:
                raise Phase3Error("A0/control capability join changed")
            paired.append(
                {
                    "capability": candidate["capability"],
                    "candidate_pass": bool(candidate["functional_pass_v1"]),
                    "teacher_pass": bool(control["functional_pass_v1"]),
                }
            )
        comparisons[system] = paired_stratified_bootstrap(
            paired, replicates=10_000, seed=5181729 + index
        )
        systems[system] = {
            "functional_passes_v1": result["functional_passes_v1"],
            "repetition_collapses_v2": result["repetition_collapses_v2"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "outputs_sha256": result["raw_outputs_sha256"],
            "evaluation_evidence_sha256": result["evidence_sha256"],
        }

    gates = {system: comparison["lower_95"] > 0.0 for system, comparison in comparisons.items()}
    result = {
        "format": "abi-capability-compiler-phase3-paired-seed-decision/1",
        "status": "PASS_ROUTE_ISOLATED_PAIRED_SEED_CAUSAL_CONTROLS" if all(gates.values()) else "FAIL_ROUTE_ISOLATED_PAIRED_SEED_CAUSAL_CONTROL_GATE",
        "protocol_sha256": protocol_sha256,
        "training_seed": int(protocol["training_seed"]),
        "A0": {
            "functional_passes_v1": a0_evaluation["functional_passes_v1"],
            "repetition_collapses_v2": a0_evaluation["repetition_collapses_v2"],
            "checkpoint_sha256": a0_checkpoint,
            "outputs_sha256": sha256_file(a0_path),
            "evaluation_evidence_sha256": a0_evaluation["evidence_sha256"],
        },
        "systems": systems,
        "paired_A0_minus_control": comparisons,
        "gates": gates,
        "all_controls_passed": all(gates.values()),
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
    result = decide(Path.cwd().resolve(), Path.cwd().resolve() / args.protocol, Path.cwd().resolve() / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
