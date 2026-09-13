"""Bind one immutable R27 source result into a LayerCake import config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import json_object, sha256_file, write_json_once
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash


def item(root: Path, path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run(heldout_config: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    source_dir = source_dir.resolve()
    result = json_object(source_dir / "result.json")
    if result.get("verdict") != "PASS" or result.get("evidence_sha256") != selfless_evidence_hash(result):
        raise ValueError("R27 source result did not pass")
    packages = []
    for record in result["packages"]:
        binding = item(root, source_dir / record["path"])
        packages.append({**binding, "namespace": record["namespace"], "facts": record["facts"]})
    value = {
        "format": "abi-r27-import-config/1", "heldout_config": item(root, heldout_config),
        "source_result": item(root, source_dir / "result.json"),
        "source_rows": item(root, source_dir / result["artifacts"]["source_rows"]["path"]),
        "evaluation": item(root, source_dir / result["artifacts"]["evaluation"]["path"]),
        "source_packages": packages,
    }
    output.parent.mkdir(parents=True, exist_ok=True); write_json_once(output, value); return value


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--heldout-config", type=Path, required=True); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); run(args.heldout_config, args.source, args.output)


if __name__ == "__main__": main()
