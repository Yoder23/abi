"""Build prospective R80 validation before R79 pointer-bridge training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog
from experiments.role_invariant_choice_r76.catalog_common import build_catalog


TRIPLES = (
    ("LIR", "MOF", "PEN"),
    ("QAZ", "RUT", "SIV"),
    ("WEX", "YUM", "ZOP"),
    ("BOL", "GEC", "HIN"),
    ("JAR", "KUM", "NEP"),
    ("PIR", "QOV", "SUT"),
    ("VEK", "WIM", "YAG"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R80 catalog exists: {args.output}")
    value = build_catalog(
        campaign="r80",
        split="validation",
        rows_per_family=200,
        numeric_offset=900_000,
        triples=TRIPLES,
        subject_prefix="NODE",
        seed_offset=80_000_000,
        status="PREREGISTERED_PROSPECTIVE_VALIDATION_ONLY",
        generator="experiments.prompt_identity_reasoning_r79.build_validation_catalog_v1",
    )
    if len(value["probes"]) != 1_400 or len({row["prompt"] for row in value["probes"]}) != 1_400:
        parser.error("R80 catalog must contain exactly 1,400 unique prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_probe_catalog(args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

