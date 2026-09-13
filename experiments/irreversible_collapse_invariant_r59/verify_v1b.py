"""Strictly recompute and optionally replay both repaired R59 screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import write_json_once
from experiments.irreversible_collapse_invariant_r59 import screen_v1
from experiments.isolated_capability_cakes_r53 import verify_v1 as common


EVIDENCE = {
    "validation": {
        "result": "154d723c6227b1dbf3b99a44e70b786906218418544c9c49a8516ad98248848a",
        "raw": "056bfe0516c59820cf8d4a4ad58d3efb702aac9dd59c987b4f2e26ece20c1934",
    },
    "final_test": {
        "result": "67fb70bc357b583eb1b1aa4e540b4e5f4ecf358c0b5e7afbda10637e21550343",
        "raw": "063ccf9b89ab1a14f559433773579256515ff55cdeeaebd658bd6ed7c5ef85f1",
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
        expected_format="abi-r59-irreversible-collapse-invariant-development-screen/1",
        expected_verdict="PASS_R59_DISCLOSED_SCREEN",
        campaign_name=f"R59_{args.split.upper()}",
    )
    write_json_once(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
