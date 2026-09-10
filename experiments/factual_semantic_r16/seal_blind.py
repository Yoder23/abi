"""Bind the clean R16 replication certificate to its exact blind review."""

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

from .seal_replication import seal_replication

REVIEWED_COMMIT = "26fb02908831cab05079ee1ba9821e401ce222d7"


def seal_blind(
    *,
    config: Path,
    reveal: Path,
    run_dir: Path,
    live: Path,
    public_requalification: Path,
    strict_path: Path,
    hostile_path: Path,
    accounting_path: Path,
    prior_certificate: Path,
    replication_certificate: Path,
    report: Path,
) -> dict[str, Any]:
    recomputed = seal_replication(
        config,
        reveal,
        run_dir,
        live,
        public_requalification,
        strict_path,
        hostile_path,
        accounting_path,
        prior_certificate,
    )
    stored = json_object(replication_certificate)
    if stored != recomputed:
        raise R14Error("R16 replication certificate changed before blind seal")
    try:
        review_text = report.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise R14Error("R16 blind report is unavailable") from exc
    required = (
        f"Reviewed commit: `{REVIEWED_COMMIT}`",
        "Bounded claim: `BOUNDED_FACTUAL_EXTRACTION_AND_SEMANTIC_SEGREGATION`",
        "Verdict: `PASS`",
        "Findings: Critical 0; High 0; Medium 0; Low 0",
        "The full ABI moonshot remains open.",
    )
    if any(value not in review_text for value in required):
        raise R14Error("R16 blind report identity or verdict changed")
    result = {
        "format": "abi-r16-bounded-blind-reviewed-certificate/1",
        "verdict": "PASS",
        "claim": stored["claim"],
        "claim_ceiling": stored["claim_ceiling"],
        "reviewed_commit": REVIEWED_COMMIT,
        "blind_findings": {"critical": 0, "high": 0, "medium": 0, "low": 0},
        "replication_certificate": {
            "path": str(replication_certificate).replace("\\", "/"),
            "sha256": sha256_file(replication_certificate),
            "evidence_sha256": stored["evidence_sha256"],
        },
        "blind_report": {
            "path": str(report).replace("\\", "/"),
            "sha256": sha256_file(report),
        },
        "publication": "PENDING",
        "independent_hardware": "PENDING",
        "full_abi_moonshot": "OPEN",
        "unproven": stored["unproven"],
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


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
    parser.add_argument("--replication-certificate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal_blind(
        config=args.config,
        reveal=args.reveal,
        run_dir=args.run_dir,
        live=args.live,
        public_requalification=args.public_requalification,
        strict_path=args.strict,
        hostile_path=args.hostile,
        accounting_path=args.accounting,
        prior_certificate=args.prior_certificate,
        replication_certificate=args.replication_certificate,
        report=args.report,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
