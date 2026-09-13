"""Run the shared fail-closed verifier against the exact R54 failure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import write_json_once
from experiments.isolated_capability_cakes_r53 import verify_v1 as common
from experiments.layerwise_sparse_control_r54 import screen_v1


EXPECTED_RESULT_SHA256 = "ad0593d13d99356bebde4ab8791e90cf20dc9476f52d0d4758f6da6f38f44773"
EXPECTED_RAW_SHA256 = "df8edaa727fd86f8d8cafd119c9f271b81bb85f6e31371b5ac268bdfbbfe3ca6"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", required=True)
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
        expected_result_sha256=EXPECTED_RESULT_SHA256,
        expected_raw_sha256=EXPECTED_RAW_SHA256,
        campaign_screen=screen_v1,
        expected_format="abi-r54-layerwise-sparse-control-development-screen/1",
        expected_verdict="FAIL_R54_DISCLOSED_SCREEN",
        campaign_name="R54",
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
