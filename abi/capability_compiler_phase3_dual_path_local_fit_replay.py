"""Exact step-zero replay after the V234 command-envelope interruption."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import capability_compiler_phase3_dual_path_local_fit as base
from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error


FORMAT = "abi-capability-compiler-phase3-dual-path-local-fit-replay/1"


def load_replay(root: Path, path: Path) -> dict:
    replay = json.loads(path.read_text(encoding="utf-8"))
    if (
        replay.get("format") != FORMAT
        or replay.get("status") != "PREREGISTERED_EXACT_STEP_ZERO_REPLAY"
        or replay.get("scientific_fields_changed") is not False
    ):
        raise Phase3Error("dual-path local-fit replay governance changed")
    for name, expected in replay["bindings"].items():
        target = Path(name) if Path(name).is_absolute() else root / name
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"dual-path local-fit replay binding changed: {name}")
    return replay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("inventory", "train"))
    parser.add_argument(
        "--protocol",
        default="ABI_CAPABILITY_COMPILER_PHASE3_DUAL_PATH_LOCAL_FIT_REPLAY_V235.json",
    )
    parser.add_argument(
        "--output",
        default="results/abi_capability_compiler_phase3_dual_path/local_fit_replay_v236",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    replay = load_replay(root, root / args.protocol)
    base_protocol = root / replay["base_scientific_protocol"]
    result = (
        base.inventory(root, base_protocol)
        if args.command == "inventory"
        else base.train(root, base_protocol, root / args.output)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not str(result["status"]).startswith("FAIL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
