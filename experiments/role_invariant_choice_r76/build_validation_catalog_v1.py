"""Build the R77 prospective catalog before R76 extraction or host training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog
from experiments.role_invariant_choice_r76.catalog_common import VALIDATION_TRIPLES, build_catalog


ROWS_PER_FAMILY = 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R77 catalog exists: {args.output}")
    value = build_catalog(
        campaign="r77",
        split="validation",
        rows_per_family=ROWS_PER_FAMILY,
        numeric_offset=700_000,
        triples=VALIDATION_TRIPLES,
        subject_prefix="ENT",
        seed_offset=77_000_000,
        status="PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        generator="experiments.role_invariant_choice_r76.build_validation_catalog_v1",
    )
    if len(value["probes"]) != 1_400 or len({row["prompt"] for row in value["probes"]}) != 1_400:
        parser.error("R77 catalog must contain exactly 1,400 unique prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

