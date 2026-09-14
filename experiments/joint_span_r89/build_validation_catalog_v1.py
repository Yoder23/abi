"""Build the R89 holdout for the already frozen R88 package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog
from experiments.role_invariant_choice_r76.catalog_common import build_catalog


TRIPLES = (
    ("ADQ", "BEX", "CYR"),
    ("DOV", "EUM", "FIK"),
    ("GOL", "HAZ", "JUP"),
    ("KEM", "LUQ", "MIR"),
    ("NEX", "OSV", "PAR"),
    ("QUM", "ROK", "SUZ"),
    ("TEQ", "VOR", "WEX"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R89 catalog exists: {args.output}")
    value = build_catalog(
        campaign="r89",
        split="validation",
        rows_per_family=200,
        numeric_offset=1_200_000,
        triples=TRIPLES,
        subject_prefix="KEY",
        seed_offset=89_000_000,
        status="PREREGISTERED_FROZEN_PACKAGE_HOLDOUT",
        generator="experiments.joint_span_r89.build_validation_catalog_v1",
    )
    probes = value["probes"]
    if (
        len(probes) != 1_400
        or len({row["prompt"] for row in probes}) != 1_400
        or any("REF" in row["prompt"] or "OBJ" in row["prompt"] for row in probes)
        or any(len(row["evaluator"]["value"][-7:]) != 7 for row in probes)
    ):
        parser.error("R89 holdout identity, token width, or coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
