"""Score R88 with the frozen method and corrected source authorization gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash
import experiments.prompt_identity_reasoning_r81.score_source_corrected_v1 as scorer


CATALOG_SHA256 = "5dba762eec849f2e6531f5f1283417d38e9470a9b37ca69cf3576bb25286a177"
SCORER_SHA256 = "54ded3d1e1afbbd016b1d4a9a837777d24bfe313241a4f9f76a4e915b4b9116c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    implementation = Path(scorer.__file__).resolve()
    if (
        hashlib.sha256(args.catalog.read_bytes()).hexdigest() != CATALOG_SHA256
        or hashlib.sha256(implementation.read_bytes()).hexdigest() != SCORER_SHA256
        or args.output.exists()
    ):
        parser.error("R88 source input, implementation, or output state changed")
    scorer.CATALOG_SHA256 = CATALOG_SHA256
    result = scorer.run(
        Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size
    )
    original_gates = dict(result.get("gates", {}))
    metrics = result.get("metrics", {})
    result["campaign"] = "r88"
    result["format"] = "abi-r88-corrected-gate-prior-corrected-source-score/1"
    result["source_family_diagnostic"] = {
        "prior_gate_value": original_gates.get("prior_corrected_family_floor"),
        "family_passing": metrics.get("prior_corrected_family_passing", {}),
        "authorization_role": "reported_nonblocking_teacher_ceiling_diagnostic",
    }
    result["gates"] = {
        "rows": metrics.get("rows") == 1_400,
        "minimum_prior_corrected_passing": metrics.get(
            "prior_corrected_passing", 0
        )
        >= 1_330,
        "all_destination_display_positions": set(
            metrics.get("prior_corrected_display_position_passing", {})
        )
        == {"0", "1", "2"},
        "zero_prior_corrected_ties": metrics.get("prior_corrected_ties") == 0,
    }
    result["verdict"] = (
        "PASS_R88_SOURCE"
        if all(result["gates"].values())
        else "FAIL_R88_SOURCE"
    )
    result["source_scoring_implementation_sha256"] = SCORER_SHA256
    result.pop("evidence_sha256", None)
    result["evidence_sha256"] = evidence_hash(result)
    result_path = args.output / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
