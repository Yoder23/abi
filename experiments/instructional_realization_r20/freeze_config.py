"""Freeze the disclosed R20 public implementation and gates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .binding import CODE_PATHS, DATA, GATES, PHYSICAL_EXTRACTION, SOURCE


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.output.exists():
        raise RuntimeError(f"immutable R20 config exists: {args.output}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    value = {
        "format": "abi-r20-public-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in CODE_PATHS},
        "source": SOURCE,
        "data": DATA,
        "gates": GATES,
        "physical_extraction": PHYSICAL_EXTRACTION,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
    print(commit)


if __name__ == "__main__":
    main()
