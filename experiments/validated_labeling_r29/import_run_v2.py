"""Path-resolution-only wrapper around the frozen R29 import implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .import_run import run as frozen_run


def run(config: Path, output: Path):
    return frozen_run(config.resolve(), output.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
