"""Verify the R76 v2 archive through the actual LayerCake consumer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.capability_compiler_phase2_common import sha256_file
from abi.capability_pipeline import verify_extraction_bundle
from abi.layercake_host_v3 import load_english_training_rows
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once


ARCHIVE_SHA256 = "292ba40ced84db5a28ef3c8214ac7645623db5e0f047218f5f7bf7c2ce0b10cc"
RECEIPT_SHA256 = "05bdf60af2f2cff4fed4b06c7e2220c506d0b9df3bc0bdf2acd8065d9255bc22"


class VerificationError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable verification exists: {args.output}")
    if sha256_file(args.archive) != ARCHIVE_SHA256 or sha256_file(args.receipt) != RECEIPT_SHA256:
        raise VerificationError("R76 v2 archive or receipt hash changed")
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    verification = verify_extraction_bundle(args.archive)
    rows, budget, bundle = load_english_training_rows(args.archive, budget_index=-1)
    ledger = bundle["ledger"]["conditional_choice_qualification"]
    if (
        verification.get("verified") is not True
        or verification.get("training_eligible") is not True
        or verification.get("domain_segregation_verified") is not True
        or len(rows) != 8_129
        or len({row["record_id"] for row in rows}) != 8_129
        or len({row["response"] for row in rows}) != 8_129
        or {row["capability"] for row in rows} != {"domain_independent_reasoning"}
        or budget.get("record_count") != 8_129
        or ledger.get("rows_scored") != 8_400
        or ledger.get("passing_rows_packaged") != 8_129
        or ledger.get("failed_rows_excluded") != 271
        or ledger.get("selective_packaging") is not True
        or receipt.get("verified") is not True
        or receipt.get("archive_sha256") != ARCHIVE_SHA256
        or receipt.get("excluded_failed_rows") != 271
    ):
        raise VerificationError("R76 v2 consumer or accounting gate failed")
    result = {
        "format": "abi-r76-selective-artifact-consumer-verification/1",
        "verdict": "PASS_R76_V2_LAYERCAKE_CONSUMER",
        "archive_sha256": ARCHIVE_SHA256,
        "receipt_sha256": RECEIPT_SHA256,
        "records": len(rows),
        "unique_responses": len({row["response"] for row in rows}),
        "source_rows": ledger["rows_scored"],
        "excluded_source_failures": ledger["failed_rows_excluded"],
        "training_eligible": verification["training_eligible"],
        "domain_segregation_verified": verification["domain_segregation_verified"],
        "selection_scope": bundle["selection"]["english_selection_scope"],
        "capabilities": sorted({row["capability"] for row in rows}),
        "layercake_candidate_trained": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

