"""Bind one completed R84 candidate before any validation generation."""

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
        parser.error(f"immutable candidate binding exists: {args.output}")
    files = {
        "checkpoint": args.candidate / "model.safetensors",
        "metadata": args.candidate / "metadata.json",
        "screen": args.screen,
        "protocol": args.protocol,
    }
    if any(not path.is_file() for path in files.values()):
        parser.error("required candidate or frozen evaluator file is absent")
    value = {
        "format": "abi-r84-candidate-screen-binding/1",
        "candidate_checkpoint_sha256": _sha256_file(files["checkpoint"]),
        "candidate_checkpoint_bytes": files["checkpoint"].stat().st_size,
        "candidate_metadata_sha256": _sha256_file(files["metadata"]),
        "candidate_metadata_bytes": files["metadata"].stat().st_size,
        "screen_sha256": _sha256_file(files["screen"]),
        "protocol_sha256": _sha256_file(files["protocol"]),
        "candidate_generation_observations_before_binding": 0,
        "full_abi_moonshot": "OPEN",
    }
    value["binding_sha256"] = _canonical_sha(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
