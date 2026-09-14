"""Fail-closed verification for the fresh R88 catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog


CATALOG_SHA256 = "5dba762eec849f2e6531f5f1283417d38e9470a9b37ca69cf3576bb25286a177"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.catalog.read_bytes()).hexdigest() != CATALOG_SHA256:
        parser.error("R88 catalog hash changed")
    probes = load_probe_catalog(args.catalog)["probes"]
    if (
        len(probes) != 1_400
        or len({row["probe_id"] for row in probes}) != 1_400
        or len({row["prompt"] for row in probes}) != 1_400
        or any(row["split"] != "validation" for row in probes)
        or any(
            row["probe_id"] != f"r88-reasoning-f{i // 200}-{i % 200:04d}"
            for i, row in enumerate(probes)
        )
        or any("REF" not in row["prompt"] for row in probes)
    ):
        parser.error("R88 catalog coverage, ordering, or identity changed")
    print(json.dumps({"status": "PASS_R88_CATALOG", "rows": len(probes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
