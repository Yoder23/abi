"""Qualify R94 with native source conditional likelihood."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash
import experiments.prompt_identity_reasoning_r81.score_source_corrected_v1 as scorer


CATALOG_SHA256 = "__FILL_AFTER_MATERIALIZATION__"
SCORER_SHA256 = "54ded3d1e1afbbd016b1d4a9a837777d24bfe313241a4f9f76a4e915b4b9116c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    implementation = Path(scorer.__file__).resolve()
    if hashlib.sha256(args.catalog.read_bytes()).hexdigest() != CATALOG_SHA256 or hashlib.sha256(implementation.read_bytes()).hexdigest() != SCORER_SHA256 or args.output.exists():
        parser.error("R94 source input, implementation, or output state changed")
    scorer.CATALOG_SHA256 = CATALOG_SHA256
    result = scorer.run(Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size)
    raw_rows = [json.loads(line) for line in (args.output / "prior_corrected_scores.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_display = Counter(str(row["destination_display_index"]) for row in raw_rows if row["raw_passed"])
    metrics = result["metrics"]
    result["campaign"] = "r94"
    result["format"] = "abi-r94-native-source-score/1"
    result["source_prior_correction_diagnostic"] = {
        "prior_corrected_passing": metrics["prior_corrected_passing"],
        "prior_corrected_family_passing": metrics["prior_corrected_family_passing"],
        "authorization_role": "reported_nonblocking_diagnostic",
    }
    result["gates"] = {
        "rows": metrics["rows"] == 1_400,
        "minimum_native_passing": metrics["raw_passing"] >= 1_330,
        "all_native_destination_display_positions": set(raw_display) == {"0", "1", "2"},
        "zero_native_ties": metrics["raw_ties"] == 0,
    }
    result["native_display_position_passing"] = dict(sorted(raw_display.items()))
    result["verdict"] = "PASS_R94_SOURCE" if all(result["gates"].values()) else "FAIL_R94_SOURCE"
    result["source_scoring_implementation_sha256"] = SCORER_SHA256
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = evidence_hash(result)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
