"""Attribute the B100 screening regression without training or final access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-b100-exposure-audit/1"
PARENT_STAGES = ("v443", "v459", "v463")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_READ_ONLY_EXPOSURE_ATTRIBUTION"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B100 exposure-audit governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B100 exposure-audit binding changed: {relative}")
    return protocol, sha256_file(path)


def audit(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable B100 exposure-audit output exists")
    b80 = {row["probe_id"]: row for row in _rows(root / protocol["b80_outputs"])}
    b100 = {row["probe_id"]: row for row in _rows(root / protocol["b100_outputs"])}
    if set(b80) != set(b100) or len(b100) != 1400:
        raise Phase3Error("B80/B100 evaluation population changed")
    regressions = [
        {
            "probe_id": probe_id,
            "capability": b100[probe_id]["capability"],
            "b80_output": b80[probe_id]["output"],
            "b100_output": b100[probe_id]["output"],
        }
        for probe_id in sorted(b100)
        if bool(b80[probe_id]["functional_pass_v1"]) and not bool(b100[probe_id]["functional_pass_v1"])
    ]
    exposure = {}
    for stage in PARENT_STAGES:
        b80_meta = _json(root / protocol["metadata"]["B80"][stage])
        b100_meta = _json(root / protocol["metadata"]["B100"][stage])
        capability = "instruction_following"
        b80_observations = int(b80_meta["training"]["sampled_records_by_capability"][capability])
        b100_observations = int(b100_meta["training"]["sampled_records_by_capability"][capability])
        b80_records = int(protocol["records_per_capability"]["B80"])
        b100_records = int(protocol["records_per_capability"]["B100"])
        exposure[stage] = {
            "B80_observations": b80_observations,
            "B100_observations": b100_observations,
            "B80_observations_per_record": b80_observations / b80_records,
            "B100_observations_per_record": b100_observations / b100_records,
            "B100_to_B80_per_record_ratio": (b100_observations / b100_records) / (b80_observations / b80_records),
        }
    exact_identifier_regressions = [row for row in regressions if row["capability"] == "instruction_following" and row["b80_output"].startswith(row["b100_output"][:3])]
    checks = {
        "all_parent_stages_reduce_per_record_exposure_to_0_8": all(abs(value["B100_to_B80_per_record_ratio"] - 0.8) < 1e-12 for value in exposure.values()),
        "instruction_following_regressions_are_identifier_omissions": len(exact_identifier_regressions) == 8 and all(row["b100_output"] in {row["b80_output"][:3] + row["b80_output"][7:]} for row in exact_identifier_regressions),
        "b100_zero_collapse": _json(root / protocol["b100_evaluation"])["repetition_collapses_v2"] == 0,
        "training_not_performed": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-b100-exposure-audit-result/1",
        "status": "PASS_EXPOSURE_DEFICIT_ATTRIBUTED_ONE_NORMALIZED_EXPOSURE_ATTEMPT_SUPPORTED" if all(checks.values()) else "FAIL_NO_SPECIFIC_SUPPORTED_INTERVENTION",
        "protocol_sha256": protocol_sha,
        "checks": checks,
        "parent_exposure": exposure,
        "B80_pass_B100_fail_rows": len(regressions),
        "identifier_omission_rows": len(exact_identifier_regressions),
        "regressions": regressions,
        "supported_intervention": "Scale only V443, V459, and V463 optimizer steps by exactly 1.25 at B100 so every parent-training record receives the B80 observation density; leave data, ordering, model, router, bridges, guard, thresholds, and evaluation unchanged.",
        "training_performed": False,
        "final_test_accessed": False,
        "claim_boundary": "Read-only attribution only; the supported intervention remains untested and no Phase 4 or superiority claim is made."
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
