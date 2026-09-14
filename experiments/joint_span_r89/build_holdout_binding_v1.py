"""Bind the frozen R88 package and complete R89 holdout before execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.conditional_choice_artifact_v2 import _canonical_sha
from abi.layercake_host import _sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate", "candidate-binding", "candidate-screen", "candidate-protocol",
        "holdout-screen", "holdout-protocol", "catalog", "source-result",
        "source-raw", "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"immutable R89 holdout binding exists: {args.output}")
    files = {
        "candidate_checkpoint": args.candidate / "bridge.safetensors",
        "candidate_metadata": args.candidate / "metadata.json",
        "candidate_binding": args.candidate_binding,
        "candidate_screen": args.candidate_screen,
        "candidate_protocol": args.candidate_protocol,
        "candidate_core": args.candidate_screen.with_name("core_v1.py"),
        "candidate_trainer": args.candidate_screen.with_name("train_candidate_v1.py"),
        "holdout_screen": args.holdout_screen,
        "holdout_protocol": args.holdout_protocol,
        "catalog": args.catalog,
        "source_result": args.source_result,
        "source_raw": args.source_raw,
    }
    if any(not path.is_file() for path in files.values()):
        parser.error("required R89 holdout input is absent")
    value = {
        "format": "abi-r89-frozen-package-holdout-binding/1",
        "holdout_observations_before_binding": 0,
        "files": {
            name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
            for name, path in sorted(files.items())
        },
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
