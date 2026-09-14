"""Capture the live frozen source on R95 without a quality admission gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash
import experiments.prompt_identity_reasoning_r81.score_source_corrected_v1 as scorer


CATALOG_SHA256 = "TO_BE_SEALED_AFTER_PREREGISTRATION"
SCORER_SHA256 = "54ded3d1e1afbbd016b1d4a9a837777d24bfe313241a4f9f76a4e915b4b9116c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    implementation = Path(scorer.__file__).resolve()
    if hashlib.sha256(args.catalog.read_bytes()).hexdigest() != CATALOG_SHA256 or hashlib.sha256(implementation.read_bytes()).hexdigest() != SCORER_SHA256 or args.output.exists():
        parser.error("R95 source input, implementation, or output state changed")
    scorer.CATALOG_SHA256 = CATALOG_SHA256
    result = scorer.run(Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size)
    rows = [json.loads(line) for line in (args.output / "prior_corrected_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_display = Counter(str(row["destination_display_index"]) for row in rows)
    metrics = result["metrics"]
    result["campaign"] = "r95"
    result["format"] = "abi-r95-live-source-capture/1"
    result["method"] = "native_mean_conditional_log_likelihood; neutral_prior_scores_retained_as_diagnostic"
    result["authorization_score"] = "capture_integrity_only"
    result["gates"] = {
        "rows": metrics["rows"] == 1_400,
        "all_destination_display_positions_captured": set(raw_display) == {"0", "1", "2"},
        "zero_native_ties": metrics["raw_ties"] == 0,
        "raw_artifact_complete": len(rows) == 1_400 and len({row["probe_id"] for row in rows}) == 1_400,
    }
    result["captured_display_position_rows"] = dict(sorted(raw_display.items()))
    result["verdict"] = "PASS_R95_SOURCE_CAPTURE" if all(result["gates"].values()) else "FAIL_R95_SOURCE_CAPTURE"
    result["source_scoring_implementation_sha256"] = SCORER_SHA256
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = evidence_hash(result)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
