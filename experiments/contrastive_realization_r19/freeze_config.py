"""Freeze the R19 held-out registration without publishing its secret."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .heldout_protocol import (
    CODE_PATHS,
    DATA,
    DEVELOPMENT_EVIDENCE,
    GATES,
    PHYSICAL_EXTRACTION,
    SOURCE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise RuntimeError(f"immutable R19 config exists: {args.output}")
    reveal = json.loads(args.reveal.read_text(encoding="utf-8"))
    if reveal.get("format") != "abi-r19-heldout-reveal/1":
        raise RuntimeError("invalid R19 reveal format")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = {
        "format": "abi-r19-heldout-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in CODE_PATHS},
        "heldout_seed_commitment": reveal["commitment"],
        "reveal_sha256": sha256_file(args.reveal),
        "development_prerequisite": {
            name: {"path": path, "sha256": sha256_file(root / path)}
            for name, path in DEVELOPMENT_EVIDENCE.items()
        },
        "source": SOURCE,
        "data": DATA,
        "gates": GATES,
        "physical_extraction": PHYSICAL_EXTRACTION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(config, indent=2, sort_keys=True).encode() + b"\n")
    print(reveal["commitment"])


if __name__ == "__main__":
    main()
