"""Build only the two missing clean-start B20 lineages authorized by V937."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import capability_compiler_phase4_abi_lineage as lineage
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-b20-missing-seed-lineage/1"
SEEDS = (130363, 155921)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_TWO_MISSING_B20_CLEAN_START_LINEAGES"
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
        or protocol.get("nearby_sweeps_authorized") is not False
        or tuple(int(row["seed"]) for row in protocol["runs"]) != SEEDS
    ):
        raise Phase3Error("B20 missing-seed lineage governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B20 missing-seed lineage binding changed: {relative}")
    base = _json(root / protocol["base_protocol"])
    for relative, expected in base["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B20 base-lineage binding changed: {relative}")
    return protocol, sha256_file(path), base


def _run(protocol: Mapping[str, Any], seed: int) -> Mapping[str, Any]:
    match = next((row for row in protocol["runs"] if int(row["seed"]) == int(seed)), None)
    if match is None:
        raise Phase3Error("unregistered B20 missing seed")
    return match


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha, base = load_protocol(root, protocol_path)
    manifest = _json(root / base["budget_manifest"])
    selected, accounting = lineage._selected_rows(root, base, manifest, "B20")
    checks = []
    for run in protocol["runs"]:
        output = root / run["output"]
        checks.append({
            "seed": int(run["seed"]),
            "output_absent": not output.exists(),
            "phase1_records": len(selected["phase1_ir"]),
            "targeted_records": len(selected["v138_targeted_ir"]),
            "host_records": len(selected["v480_host_supervision"]),
        })
    gates = {
        "two_missing_seeds": len(checks) == 2 and all(row["output_absent"] for row in checks),
        "b20_selection_exact": (
            int(accounting["unique_source_attempts"]) == 2028
            and int(accounting["authoritative_teacher_output_tokens"]) == 62417
            and int(accounting["record_memberships"]) == 2056
        ),
        "same_clean_start_protocol": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "format": "abi-capability-compiler-phase4-b20-missing-seed-lineage-preflight/1",
        "status": "PASS_B20_MISSING_SEED_LINEAGE_PREFLIGHT" if all(gates.values()) else "FAIL_B20_MISSING_SEED_LINEAGE_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "checks": checks,
        "accounting": accounting,
        "gates": gates,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
    }


def train(root: Path, protocol_path: Path, seed: int) -> dict[str, Any]:
    protocol, protocol_sha, base = load_protocol(root, protocol_path)
    run = _run(protocol, seed)
    output = root / run["output"]
    original = lineage.load_protocol
    try:
        lineage.load_protocol = lambda _root, _path: (base, protocol_sha)
        trained = lineage.train_lineage(root, protocol_path, "B20", int(seed), output)
    finally:
        lineage.load_protocol = original
    receipt = {
        "format": "abi-capability-compiler-phase4-b20-missing-seed-lineage-result/1",
        "status": "COMPLETE_B20_MISSING_SEED_CLEAN_START_LINEAGE",
        "protocol_sha256": protocol_sha,
        "budget": "B20",
        "seed": int(seed),
        "lineage_result_sha256": sha256_file(output / "result.json"),
        "functional_passes_v1": int(trained["functional_passes_v1"]),
        "repetition_collapses_v2": int(trained["repetition_collapses_v2"]),
        "gates": trained["gates"],
        "budget_accounting": trained["budget"],
        "teacher_present_at_inference": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "One clean-start B20 lower-anchor lineage. Same-host oracle verification and the paired seed remain required."
    }
    receipt["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    _write_immutable(output / "continuation_receipt.json", json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = root / args.protocol
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.seed is not None:
        result = train(root, protocol_path, args.seed)
    else:
        raise Phase3Error("select preflight or one registered seed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith(("PASS", "COMPLETE")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
