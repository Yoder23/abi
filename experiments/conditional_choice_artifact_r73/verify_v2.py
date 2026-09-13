"""Verify R73-v2 and its complete LayerCake training-consumer contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_source_artifact import _canonical_sha
from abi.layercake_host_v3 import load_english_training_rows
from experiments.conditional_choice_artifact_r73 import verify_v1 as base
from experiments.foreign_capability_r14.core import write_json_once


ARCHIVE_SHA256 = "3450f510430ab4402b9a052d1a6b928f594bf2b759abc6e3c5baa872e8758b54"
RECEIPT_FILE_SHA256 = "5c322d737882cb7c3dfd463818a7b3db0486c4e374ec341120c45f229e9b8b1a"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--raw-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    base.EXPECTED_ARCHIVE_SHA256 = ARCHIVE_SHA256
    base.EXPECTED_RECEIPT_FILE_SHA256 = RECEIPT_FILE_SHA256
    value = base.verify(
        artifact_path=args.artifact,
        receipt_path=args.receipt,
        catalog_path=args.catalog,
        result_path=args.result,
        raw_scores_path=args.raw_scores,
    )
    rows, budget, bundle = load_english_training_rows(args.artifact, budget_index=-1)
    if (
        len(rows) != 2_098
        or {row["capability"] for row in rows}
        != {"domain_independent_reasoning"}
        or bundle["selection"]["requested_english_capabilities"]
        != ["domain_independent_reasoning"]
        or budget["record_count"] != 2_098
    ):
        raise RuntimeError("R73-v2 complete consumer contract failed")
    value.update(
        {
            "format": "abi-conditional-choice-source-artifact-verification/2",
            "verdict": "PASS_R73_V2_COMPLETE_TRAINING_CONSUMER",
            "archive_sha256": ARCHIVE_SHA256,
            "receipt_file_sha256": RECEIPT_FILE_SHA256,
            "complete_training_consumer_rows": len(rows),
            "requested_english_capabilities": ["domain_independent_reasoning"],
            "v1_training_authorized": False,
        }
    )
    value.pop("evidence_sha256", None)
    value["evidence_sha256"] = _canonical_sha(value)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
