"""Fail-closed verification for the fresh R85 catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from abi.hf_extraction import load_probe_catalog


CATALOG_SHA256 = "ed2876026551cac8243fe4a15a78e884de913d31dda181524b44553fbc0b7c6d"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.catalog.read_bytes()).hexdigest() != CATALOG_SHA256:
        parser.error("R85 catalog hash changed")
    value = load_probe_catalog(args.catalog)
    probes = value["probes"]
    if (
        len(probes) != 1_400
        or len({row["probe_id"] for row in probes}) != 1_400
        or len({row["prompt"] for row in probes}) != 1_400
        or any(row["split"] != "validation" for row in probes)
        or any(row["probe_id"] != f"r85-reasoning-f{i // 200}-{i % 200:04d}" for i, row in enumerate(probes))
    ):
        parser.error("R85 catalog coverage or ordering changed")
    print(json.dumps({"status": "PASS_R85_CATALOG", "rows": len(probes)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
