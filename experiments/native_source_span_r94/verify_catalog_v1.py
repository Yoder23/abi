"""Verify R94 catalog identity, shape, and disjointness."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog


CATALOG_SHA256 = "__FILL_AFTER_MATERIALIZATION__"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    args = parser.parse_args()
    catalog = args.catalog.resolve()
    if CATALOG_SHA256.startswith("__") or hashlib.sha256(catalog.read_bytes()).hexdigest() != CATALOG_SHA256:
        parser.error("R94 catalog is unsealed or changed")
    probes = list(load_probe_catalog(catalog)["probes"])
    prompts = {str(row["prompt"]) for row in probes}
    old = set()
    for path in args.search_root.resolve().rglob("*.json"):
        if path.resolve() == catalog:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("probes"), list):
            old.update(str(row["prompt"]) for row in value["probes"] if isinstance(row, dict) and "prompt" in row)
    family = {str(index): 0 for index in range(4)}
    placement = {str(index): 0 for index in range(4)}
    for index, row in enumerate(probes):
        expected = index // 350
        if row["probe_id"] != f"r94-reasoning-f{expected}-{index % 350:04d}":
            parser.error("R94 row identity changed")
        family[str(row["structural_family"])] += 1
        placement[str(row["candidate_list_placement"])] += 1
    if len(probes) != 1_400 or len(prompts) != 1_400 or prompts & old or set(family.values()) != {350} or set(placement.values()) != {350}:
        parser.error("R94 shape or disjointness failed")
    print(json.dumps({
        "format": "abi-r94-catalog-verification/1", "verdict": "PASS_R94_CATALOG",
        "catalog_sha256": CATALOG_SHA256, "rows": 1_400, "unique_prompts": 1_400,
        "exact_prior_prompt_overlap": 0, "family_counts": family,
        "candidate_list_placement_counts": placement,
        "candidate_loaded": False, "teacher_loaded": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
