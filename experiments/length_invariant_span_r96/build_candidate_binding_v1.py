"""Freeze R96 checkpoint identity before any candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import write_json_once
from experiments.length_invariant_span_r96.core_v1 import load_package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    _, metadata = load_package(args.candidate, device="cpu")
    files = {
        "checkpoint": args.candidate / "bridge.safetensors",
        "metadata": args.candidate / "metadata.json",
        "core": Path(__file__).with_name("core_v1.py"),
        "trainer": Path(__file__).with_name("train_v1.py"),
        "protocol": Path(__file__).with_name("PROTOCOL.md"),
    }
    value = {
        "format": "abi-r96-frozen-candidate-binding/1",
        "candidate_generation_observations_before_binding": 0,
        "r95_outputs_used_for_training": metadata["development"]["r95_outputs_used_for_training"],
        "files": {name: {
            "path": str(path.resolve().relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        } for name, path in files.items()},
        "full_abi_moonshot": "OPEN",
        "promotion_eligible": False,
    }
    value["binding_sha256"] = _canonical_sha(value)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
