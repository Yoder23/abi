"""Exact runtime replay for V343 after the preserved Linear.dtype defect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import capability_compiler_phase3_routed_v16_trajectory_retargeting as original


def execute(root: Path, protocol_path: Path, output: Path) -> dict:
    if hasattr(torch.nn.Linear, "dtype"):
        raise RuntimeError("unexpected pre-existing Linear.dtype attribute")
    torch.nn.Linear.dtype = property(lambda module: module.weight.dtype)
    try:
        return original.execute(root, protocol_path, output)
    finally:
        delattr(torch.nn.Linear, "dtype")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_ROUTED_V16_TRAJECTORY_RETARGETING_REPLAY_PROTOCOL_V345.json")
    parser.add_argument("--output-dir", default="results/abi_capability_compiler_phase3_routed_v16/trajectory_retargeting_v346")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    print(json.dumps(execute(root, (root / args.protocol).resolve(), (root / args.output_dir).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
