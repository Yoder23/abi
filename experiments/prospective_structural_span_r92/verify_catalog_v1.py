"""Verify R92 shape, identities, and disjointness before model access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog


CATALOG_SHA256 = "ae5686361a74f1539571f03568883532b17d139c65e77ba245cdea67b1ee6fee"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--search-root", required=True, type=Path)
    args = parser.parse_args()
    catalog = args.catalog.resolve()
    if hashlib.sha256(catalog.read_bytes()).hexdigest() != CATALOG_SHA256:
        parser.error("R92 catalog hash changed")
    probes = list(load_probe_catalog(catalog)["probes"])
    prompts = {str(row["prompt"]) for row in probes}
    old_prompts = set()
    for path in args.search_root.resolve().rglob("*.json"):
        if path.resolve() == catalog:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and isinstance(value.get("probes"), list):
            old_prompts.update(str(row.get("prompt")) for row in value["probes"] if isinstance(row, dict) and "prompt" in row)
    family = {str(index): 0 for index in range(7)}
    placement = {str(index): 0 for index in range(4)}
    for index, row in enumerate(probes):
        expected_family = index // 200
        if row["probe_id"] != f"r92-reasoning-f{expected_family}-{index % 200:04d}":
            parser.error("R92 row order or identity changed")
        family[str(row["structural_family"])] += 1
        placement[str(row["candidate_list_placement"])] += 1
    result = {
        "format": "abi-r92-prospective-catalog-verification/1",
        "verdict": "PASS_R92_CATALOG",
        "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
        "rows": len(probes), "unique_prompts": len(prompts),
        "exact_prior_prompt_overlap": len(prompts & old_prompts),
        "family_counts": family, "candidate_list_placement_counts": placement,
        "candidate_loaded": False, "teacher_loaded": False,
    }
    if (
        result["rows"] != 1_400 or result["unique_prompts"] != 1_400
        or result["exact_prior_prompt_overlap"] != 0
        or set(family.values()) != {200} or set(placement.values()) != {350}
    ):
        parser.error("R92 catalog shape or disjointness failed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
