"""Build a fresh prospective R85 catalog before source or candidate scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog
from experiments.role_invariant_choice_r76.catalog_common import build_catalog


TRIPLES = (
    ("ALP", "BRI", "CEN"),
    ("DAX", "ERI", "FON"),
    ("GAV", "HIR", "IVO"),
    ("JEX", "KAL", "LOR"),
    ("MUN", "NAV", "ORP"),
    ("QES", "RUD", "SIV"),
    ("TAW", "URO", "WIN"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R85 catalog exists: {args.output}")
    value = build_catalog(
        campaign="r85",
        split="validation",
        rows_per_family=200,
        numeric_offset=970_000,
        triples=TRIPLES,
        subject_prefix="NODE",
        seed_offset=85_000_000,
        status="PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        generator="experiments.stateful_span_r85.build_validation_catalog_v1",
    )
    probes = value["probes"]
    if (
        len(probes) != 1_400
        or len({row["prompt"] for row in probes}) != 1_400
        or any("VERTEX" in row["prompt"] for row in probes)
    ):
        parser.error("R85 fresh-catalog identity or coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
