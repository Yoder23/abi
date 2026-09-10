"""Seal R16 from recomputed strict, live, and hostile evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .hostile_audit import audit
from .verify import verify


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if value.get("evidence_sha256") != evidence_hash(scientific):
        raise R14Error(f"R16 {name} evidence hash changed")
    return value


def seal(
    config: Path,
    reveal: Path,
    run_dir: Path,
    strict_path: Path,
    live_path: Path,
    hostile_path: Path,
) -> dict[str, Any]:
    recomputed_strict = verify(config, reveal, run_dir)
    stored_strict = _verified(strict_path, "strict receipt")
    if recomputed_strict != stored_strict:
        raise R14Error("R16 stored strict receipt changed")
    live = _verified(live_path, "live receipt")
    if live.get("verdict") != "PASS" or live.get("physical_extractions_replayed") != 2:
        raise R14Error("R16 live gate changed")
    stored_hostile = _verified(hostile_path, "hostile receipt")
    if stored_hostile != audit(config, reveal, run_dir):
        raise R14Error("R16 hostile audit changed")
    run_receipt = _verified(run_dir / "receipt.json", "run receipt")
    certificate = {
        "format": "abi-r16-bounded-certificate/1",
        "verdict": "PASS",
        "claim": recomputed_strict["claim"],
        "claim_ceiling": recomputed_strict["claim_ceiling"],
        "full_abi_moonshot": "OPEN",
        "results": {
            "facts_exact": recomputed_strict["facts_exact"],
            "evaluation_rows_exact": recomputed_strict["evaluation_rows_exact"],
            "namespaces": recomputed_strict["namespaces"],
            "packages": len(run_receipt["packages"]),
            "source_training_steps": 0,
            "teacher_present_at_package_execution": False,
            "hostile_cases_rejected": stored_hostile["cases_passed"],
        },
        "information_accounting": run_receipt["information_accounting"],
        "evidence": {
            "run_sha256": sha256_file(run_dir / "receipt.json"),
            "strict_sha256": sha256_file(strict_path),
            "live_sha256": sha256_file(live_path),
            "hostile_sha256": sha256_file(hostile_path),
        },
        "unproven": run_receipt["unproven"],
    }
    certificate["evidence_sha256"] = evidence_hash(certificate)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--hostile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.config, args.reveal, args.run_dir, args.strict, args.live, args.hostile)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
