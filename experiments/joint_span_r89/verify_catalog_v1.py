"""Fail-closed verification for the frozen R89 package holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog


CATALOG_SHA256 = "e21bf7e3a2f3c72d5b82307009fb05c3e1e0623208d9023a13baeaf1e265a118"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.catalog.read_bytes()).hexdigest() != CATALOG_SHA256:
        parser.error("R89 catalog hash changed")
    probes = load_probe_catalog(args.catalog)["probes"]
    if (
        len(probes) != 1_400
        or len({row["probe_id"] for row in probes}) != 1_400
        or len({row["prompt"] for row in probes}) != 1_400
        or any(row["split"] != "validation" for row in probes)
        or any(
            row["probe_id"] != f"r89-reasoning-f{i // 200}-{i % 200:04d}"
            for i, row in enumerate(probes)
        )
        or any("KEY" not in row["prompt"] for row in probes)
        or any(len(row["evaluator"]["value"][-7:]) != 7 for row in probes)
    ):
        parser.error("R89 catalog coverage, ordering, or token width changed")
    print(json.dumps({"status": "PASS_R89_CATALOG", "rows": len(probes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
