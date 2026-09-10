"""Final local R16 seal including expanded information accounting."""

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

from .accounting import account
from .hostile_audit_v2 import audit_v2
from .verify_strict_v2 import verify_v2


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if value.get("evidence_sha256") != evidence_hash(scientific):
        raise R14Error(f"R16 {name} evidence hash changed")
    return value


def seal_v3(
    config: Path,
    reveal: Path,
    run_dir: Path,
    strict_path: Path,
    live_path: Path,
    hostile_path: Path,
    accounting_path: Path,
) -> dict[str, Any]:
    strict = verify_v2(config, reveal, run_dir, live_path)
    if strict != _verified(strict_path, "strict-v2 receipt"):
        raise R14Error("R16 strict-v2 receipt changed")
    hostile = _verified(hostile_path, "hostile-v2 receipt")
    if hostile != audit_v2(config, reveal, run_dir, live_path):
        raise R14Error("R16 hostile-v2 receipt changed")
    accounting = _verified(accounting_path, "accounting receipt")
    if accounting != account(run_dir):
        raise R14Error("R16 accounting receipt changed")
    receipt = _verified(run_dir / "receipt.json", "run receipt")
    certificate = {
        "format": "abi-r16-bounded-certificate/3",
        "repair_of": "results/factual_semantic_r16/heldout_v1_certificate_v2.json",
        "verdict": "PASS",
        "claim": strict["claim"],
        "claim_ceiling": strict["claim_ceiling"],
        "full_abi_moonshot": "OPEN",
        "chronology": {
            "implementation_freeze_commit": "b1a3904",
            "preregistration_commit": "b8e97ba",
            "reveal_commit": "bd5e571",
        },
        "results": {
            "facts_exact": strict["facts_exact"],
            "evaluation_rows_exact": strict["evaluation_rows_exact"],
            "namespaces": strict["namespaces"],
            "packages": len(receipt["packages"]),
            "source_training_steps": 0,
            "teacher_present_at_package_execution": False,
            "hostile_cases_rejected": hostile["cases_passed"],
            "declared_artifacts_verified": strict["declared_artifacts_verified"],
            "source_residual_rows_verified": strict["source_residual_rows_verified"],
            "live_files_verified": strict["live_files_verified"],
        },
        "information_accounting": accounting,
        "source_snapshot_evidence_sha256": strict["source_snapshot_evidence_sha256"],
        "evidence": {
            "run_sha256": sha256_file(run_dir / "receipt.json"),
            "strict_v2_sha256": sha256_file(strict_path),
            "live_sha256": sha256_file(live_path),
            "hostile_v2_sha256": sha256_file(hostile_path),
            "accounting_sha256": sha256_file(accounting_path),
        },
        "unproven": receipt["unproven"],
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
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal_v3(
        args.config,
        args.reveal,
        args.run_dir,
        args.strict,
        args.live,
        args.hostile,
        args.accounting,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
