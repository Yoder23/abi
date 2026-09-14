"""Qualify the pinned source on R93 before candidate access."""

from __future__ import annotations

import argparse
import hashlib
import json
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
        parser.error("R93 source input, implementation, or output state changed")
    scorer.CATALOG_SHA256 = CATALOG_SHA256
    result = scorer.run(Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size)
    metrics = result.get("metrics", {})
    result["campaign"] = "r93"
    result["format"] = "abi-r93-prior-corrected-source-score/1"
    result["source_family_diagnostic"] = {
        "family_passing": metrics.get("prior_corrected_family_passing", {}),
        "authorization_role": "reported_nonblocking_teacher_ceiling_diagnostic",
    }
    result["gates"] = {
        "rows": metrics.get("rows") == 1_400,
        "minimum_prior_corrected_passing": metrics.get("prior_corrected_passing", 0) >= 1_330,
        "all_destination_display_positions": set(metrics.get("prior_corrected_display_position_passing", {})) == {"0", "1", "2"},
        "zero_prior_corrected_ties": metrics.get("prior_corrected_ties") == 0,
    }
    result["verdict"] = "PASS_R93_SOURCE" if all(result["gates"].values()) else "FAIL_R93_SOURCE"
    result["source_scoring_implementation_sha256"] = SCORER_SHA256
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = evidence_hash(result)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
