"""Run fail-closed raw recomputation and live replay for R55."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.deep_sparse_adapters_r55 import screen_v1
from experiments.foreign_capability_r14.core import write_json_once
from experiments.isolated_capability_cakes_r53 import verify_v1 as common


EVIDENCE = {
    "validation": {
        "result": "7222448963e2496049c0483213bbe6a10e0790468421474a92051b4cf915c3e5",
        "raw": "426f0885ea0dfea5bbd9054729ff45543d4f17ba857250c7bd67a625631913ec",
        "verdict": "PASS_R55_DISCLOSED_SCREEN",
    },
    "final_test": {
        "result": "e15fc26d42587111efa82b28bdd4a11bb1a27083e89776ae2206332048b0d70e",
        "raw": "a4f65940a85f8d6d21ea032e53a55dfd2f40c9c18d711319df6d108cf1203866",
        "verdict": "FAIL_R55_DISCLOSED_SCREEN",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EVIDENCE), required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", default=[])
    parser.add_argument("--live-source-dir", type=Path)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    expected = EVIDENCE[args.split]
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
        split=args.split,
        live_source_dir=(
            args.live_source_dir.resolve() if args.live_source_dir else None
        ),
        expected_result_sha256=expected["result"],
        expected_raw_sha256=expected["raw"],
        campaign_screen=screen_v1,
        expected_format="abi-r55-deep-sparse-adapters-development-screen/1",
        expected_verdict=expected["verdict"],
        campaign_name=f"R55_{args.split.upper()}",
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
