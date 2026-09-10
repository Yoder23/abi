"""Seal the clean R16 held-out replication after provenance repair."""

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

from .accounting_v3 import account_v3
from .hostile_audit_v4 import audit_v4
from .verify_strict_v4 import verify_v4

IMPLEMENTATION_FREEZE = "a3255e0dfdb4b61b6dc344e5aa3c6c421662873b"
PREREGISTRATION = "ec765cf35a4652cb8dac1bb48d5904ca9dee19b3"
REVEAL = "4eb64ba432661bbf349be8c8bc8388d6e8d812f5"


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    scientific = {key: item for key, item in value.items() if key != "evidence_sha256"}
    if value.get("evidence_sha256") != evidence_hash(scientific):
        raise R14Error(f"R16 {name} evidence hash changed")
    return value


def seal_replication(
    config: Path,
    reveal: Path,
    run_dir: Path,
    live: Path,
    public_requalification: Path,
    strict_path: Path,
    hostile_path: Path,
    accounting_path: Path,
    prior_certificate: Path,
) -> dict[str, Any]:
    strict = _verified(strict_path, "replication strict receipt")
    if strict != verify_v4(config, reveal, run_dir, live, public_requalification):
        raise R14Error("R16 replication strict receipt changed")
    hostile = _verified(hostile_path, "replication hostile receipt")
    if hostile != audit_v4(config, reveal, run_dir, live, public_requalification):
        raise R14Error("R16 replication hostile receipt changed")
    accounting = _verified(accounting_path, "replication accounting")
    if accounting != account_v3(run_dir):
        raise R14Error("R16 replication accounting changed")
    prior = _verified(prior_certificate, "prior bounded certificate")
    run = _verified(run_dir / "receipt.json", "replication run receipt")
    config_value = json_object(config)
    if (
        config_value.get("replication", {}).get("implementation_freeze_commit")
        != IMPLEMENTATION_FREEZE
        or config_value["public_prerequisite"]["receipt"]
        != "results/factual_semantic_r16/public_v2_requalification_003/receipt.json"
        or config_value["public_prerequisite"]["sha256"]
        != sha256_file(public_requalification / "receipt.json")
    ):
        raise R14Error("R16 replication chronology or public prerequisite changed")
    certificate = {
        "format": "abi-r16-bounded-replication-certificate/1",
        "verdict": "PASS",
        "claim": strict["claim"],
        "claim_ceiling": strict["claim_ceiling"],
        "full_abi_moonshot": "OPEN",
        "replication_of": str(prior_certificate).replace("\\", "/"),
        "chronology": {
            "implementation_freeze_commit": IMPLEMENTATION_FREEZE,
            "preregistration_commit": PREREGISTRATION,
            "reveal_commit": REVEAL,
        },
        "results": {
            "facts_exact": strict["facts_exact"],
            "evaluation_rows_exact": strict["evaluation_rows_exact"],
            "namespaces": strict["namespaces"],
            "packages": len(run["packages"]),
            "package_bytes": sum(int(item["bytes"]) for item in run["packages"]),
            "hostile_cases_rejected": hostile["cases_passed"],
            "actual_live_files_rehashed": strict["actual_live_files_rehashed"],
            "physical_extraction_trees_verified": strict[
                "physical_extraction_trees_verified"
            ],
            "source_snapshot_files_rehashed": strict["source_snapshot_files_rehashed"],
            "source_training_steps": 0,
            "teacher_present_at_package_execution": False,
        },
        "candidate_boundary": strict["candidate_boundary"],
        "information_accounting": accounting,
        "packages": run["packages"],
        "evidence": {
            "config_sha256": sha256_file(config),
            "reveal_sha256": sha256_file(reveal),
            "run_receipt_sha256": sha256_file(run_dir / "receipt.json"),
            "live_receipt_sha256": sha256_file(live),
            "strict_v4_sha256": sha256_file(strict_path),
            "hostile_v4_sha256": sha256_file(hostile_path),
            "accounting_v3_sha256": sha256_file(accounting_path),
            "public_requalification_receipt_sha256": sha256_file(
                public_requalification / "receipt.json"
            ),
            "prior_certificate_sha256": sha256_file(prior_certificate),
            "prior_certificate_evidence_sha256": prior["evidence_sha256"],
        },
        "unproven": run["unproven"],
    }
    certificate["evidence_sha256"] = evidence_hash(certificate)
    return certificate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--public-requalification", type=Path, required=True)
    parser.add_argument("--strict", type=Path, required=True)
    parser.add_argument("--hostile", type=Path, required=True)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--prior-certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal_replication(
        args.config,
        args.reveal,
        args.run_dir,
        args.live,
        args.public_requalification,
        args.strict,
        args.hostile,
        args.accounting,
        args.prior_certificate,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
