"""Bind the fresh R26 R16 artifacts to the unchanged R25 engine schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file


def _binding(root: Path, relative: str, **extra: object) -> dict[str, object]:
    target = root / relative
    return {
        "path": relative,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--source-strict", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R26 import config")
    root = Path(__file__).resolve().parents[2]
    base = json_object(
        root / "experiments/canonical_layercake_import_r25/configs/public_v1.json"
    )
    source_run = root / args.source_run
    receipt = json_object(source_run / "receipt.json")
    base["r16_source_rows"] = _binding(
        root, (args.source_run / "source_observations.jsonl").as_posix()
    )
    base["r16_receipt"] = _binding(
        root, (args.source_run / "receipt.json").as_posix()
    )
    base["r16_strict_verification"] = _binding(
        root, args.source_strict.as_posix()
    )
    base["r16_packages"] = [
        _binding(
            root,
            (args.source_run / "extraction" / item["path"]).as_posix(),
            namespace=item["namespace"],
            facts=item["facts"],
        )
        for item in receipt["packages"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(base, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
