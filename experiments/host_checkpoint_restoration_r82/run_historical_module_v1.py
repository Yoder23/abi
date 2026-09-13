"""Run an untouched archived ABI Python package against the live evidence root."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


class HistoricalLauncherError(RuntimeError):
    pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    snapshot = args.snapshot.resolve()
    restoration_root = (root / "results" / "host_checkpoint_restoration_r82").resolve()
    try:
        snapshot.relative_to(restoration_root)
    except ValueError as exc:
        raise HistoricalLauncherError("snapshot must be inside the R82 evidence root") from exc
    module_path = snapshot / "abi" / "layercake_full_core_acquisition.py"
    if not module_path.is_file():
        raise HistoricalLauncherError("historical acquisition module is absent")
    if "abi" in sys.modules:
        raise HistoricalLauncherError("ABI package was imported before snapshot isolation")
    historical_arguments = list(args.arguments)
    if historical_arguments[:1] == ["--"]:
        historical_arguments = historical_arguments[1:]
    sys.path.insert(0, str(snapshot))
    sys.argv = [str(module_path), *historical_arguments]
    runpy.run_module("abi.layercake_full_core_acquisition", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
