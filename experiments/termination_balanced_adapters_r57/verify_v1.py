"""Fail closed while recomputing and optionally replaying R57 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import write_json_once
from experiments.isolated_capability_cakes_r53 import verify_v1 as common
from experiments.termination_balanced_adapters_r57 import screen_v1


RESULT_SHA256 = "ce21fad2c1f5596ac708bce69d7d7c294fc779320b6510dbed2bcb54788b52da"
RAW_SHA256 = "477cc03797b3513962fbdfc1c523a2da564c97dec6418dd7ea22c2ec80b9dbff"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", default=[])
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    receipt = common.verify(
        args.run_dir.resolve(),
        args.candidate.resolve(),
        args.router.resolve(),
        args.catalog.resolve(),
        [path.resolve() for path in args.source_bundle],
        args.broad_bundle.resolve(),
        args.anchor_bundle.resolve(),
        args.layercake_root.resolve(),
        args.live,
        split="validation",
        live_source_dir=None,
        expected_result_sha256=RESULT_SHA256,
        expected_raw_sha256=RAW_SHA256,
        campaign_screen=screen_v1,
        expected_format="abi-r57-terminal-balanced-adapters-development-screen/1",
        expected_verdict="FAIL_R57_DISCLOSED_SCREEN",
        campaign_name="R57_VALIDATION",
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
