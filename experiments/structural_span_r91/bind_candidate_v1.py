"""Bind the trained R91 candidate before its disclosed development screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file
from experiments.foreign_capability_r14.core import write_json_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    candidate = args.candidate.resolve()
    files = {
        "core_sha256": root / "experiments/structural_span_r91/core_v1.py",
        "trainer_sha256": root / "experiments/structural_span_r91/train_v1.py",
        "screen_sha256": root / "experiments/structural_span_r91/screen_development_v1.py",
        "protocol_sha256": root / "experiments/structural_span_r91/PROTOCOL.md",
    }
    value = {
        "format": "abi-r91-frozen-development-candidate-binding/1",
        "candidate_checkpoint_sha256": _sha256_file(candidate / "bridge.safetensors"),
        "candidate_checkpoint_bytes": (candidate / "bridge.safetensors").stat().st_size,
        "candidate_metadata_sha256": _sha256_file(candidate / "metadata.json"),
        "candidate_metadata_bytes": (candidate / "metadata.json").stat().st_size,
        "candidate_generation_observations_before_binding": 0,
        "r60_outputs_used_for_training": 0,
        "development_catalog_sha256": "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7",
        "development_raw_sha256": "e509484babfab685c91fee3a6165bfea0b2881213fdb7b3a71fde9266e2a25cf",
        **{name: _sha256_file(path) for name, path in files.items()},
        "promotion_eligible": False,
        "full_abi_moonshot": "OPEN",
    }
    value["binding_sha256"] = _canonical_sha(value)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
