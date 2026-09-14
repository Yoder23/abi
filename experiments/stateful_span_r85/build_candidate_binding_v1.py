"""Bind the R85 span package before any candidate generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--screen", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R85 binding exists: {args.output}")
    files = {
        "checkpoint": args.candidate / "bridge.safetensors",
        "metadata": args.candidate / "metadata.json",
        "screen": args.screen,
        "protocol": args.protocol,
        "core": args.screen.with_name("core_v1.py"),
        "trainer": args.screen.with_name("train_candidate_v1.py"),
    }
    if any(not path.is_file() for path in files.values()):
        parser.error("required R85 package or implementation file is absent")
    value = {
        "format": "abi-r85-span-candidate-binding/1",
        "candidate_checkpoint_sha256": _sha256_file(files["checkpoint"]),
        "candidate_checkpoint_bytes": files["checkpoint"].stat().st_size,
        "candidate_metadata_sha256": _sha256_file(files["metadata"]),
        "candidate_metadata_bytes": files["metadata"].stat().st_size,
        "screen_sha256": _sha256_file(files["screen"]),
        "protocol_sha256": _sha256_file(files["protocol"]),
        "core_sha256": _sha256_file(files["core"]),
        "trainer_sha256": _sha256_file(files["trainer"]),
        "candidate_generation_observations_before_binding": 0,
        "full_abi_moonshot": "OPEN",
    }
    value["binding_sha256"] = _canonical_sha(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
