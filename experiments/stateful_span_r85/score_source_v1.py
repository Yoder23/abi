"""Score the fresh R85 catalog with the frozen R81 source method."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash
import experiments.prompt_identity_reasoning_r81.score_source_corrected_v1 as scorer


CATALOG_SHA256 = "ed2876026551cac8243fe4a15a78e884de913d31dda181524b44553fbc0b7c6d"
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
        parser.error("R85 source input, implementation, or output state changed")
    scorer.CATALOG_SHA256 = CATALOG_SHA256
    result = scorer.run(
        Path.cwd(), args.catalog.resolve(), args.output.resolve(), args.batch_size
    )
    result["campaign"] = "r85"
    result["format"] = "abi-r85-prospective-prior-corrected-source-score/1"
    result["verdict"] = (
        "PASS_R85_SOURCE"
        if all(result.get("gates", {}).values())
        else "FAIL_R85_SOURCE"
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
