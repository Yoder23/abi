"""Build the fresh prospective R86 catalog before source or candidate scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog
from experiments.role_invariant_choice_r76.catalog_common import build_catalog


TRIPLES = (
    ("BEQ", "CUV", "DIP"),
    ("FAR", "GES", "HIN"),
    ("JOM", "KUR", "LEV"),
    ("MOQ", "NUX", "PIR"),
    ("RAS", "TEV", "WOK"),
    ("XEL", "YIM", "ZOR"),
    ("BOV", "CIR", "DUX"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R86 catalog exists: {args.output}")
    value = build_catalog(
        campaign="r86",
        split="validation",
        rows_per_family=200,
        numeric_offset=980_000,
        triples=TRIPLES,
        subject_prefix="SUB",
        seed_offset=86_000_000,
        status="PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        generator="experiments.joint_span_r86.build_validation_catalog_v1",
    )
    probes = value["probes"]
    if (
        len(probes) != 1_400
        or len({row["prompt"] for row in probes}) != 1_400
        or any("NODE" in row["prompt"] or "OBJ" in row["prompt"] for row in probes)
    ):
        parser.error("R86 fresh-catalog identity or coverage changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
