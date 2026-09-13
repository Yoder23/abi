"""Build the prospective R81 validation catalog before source scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog
from experiments.role_invariant_choice_r76.catalog_common import build_catalog


TRIPLES = (
    ("BEX", "CIJ", "DOV"),
    ("FAQ", "GUM", "HES"),
    ("KOJ", "LUX", "MIV"),
    ("NUQ", "PEZ", "RAK"),
    ("SOX", "TUB", "VIF"),
    ("WOQ", "XER", "YIB"),
    ("ZAC", "BEV", "CUK"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R81 catalog exists: {args.output}")
    value = build_catalog(
        campaign="r81",
        split="validation",
        rows_per_family=200,
        numeric_offset=960_000,
        triples=TRIPLES,
        subject_prefix="VERTEX",
        seed_offset=81_000_000,
        status="PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        generator="experiments.prompt_identity_reasoning_r81.build_validation_catalog_v1",
    )
    if len(value["probes"]) != 1_400 or len({row["prompt"] for row in value["probes"]}) != 1_400:
        parser.error("R81 catalog must contain exactly 1,400 unique prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
