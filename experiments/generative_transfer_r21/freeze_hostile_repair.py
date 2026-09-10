"""Preregister the R21 targeted hostile-control repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .hostile_repair_binding import HOSTILE_REPAIR_CODE_PATHS


def _binding(root: Path, relative: str) -> dict[str, object]:
    target = root / relative
    return {
        "path": relative,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    config = {
        "format": "abi-r21-hostile-repair-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in HOSTILE_REPAIR_CODE_PATHS
        },
        "live_verification_config": _binding(
            root, "experiments/generative_transfer_r21/configs/live_verification_v6.json"
        ),
        "failed_verifier_record": _binding(
            root,
            "results/generative_transfer_r21/public_v6_live_verification/verifier_failure.json",
        ),
        "old_mutation": "final_tar_padding_byte",
        "new_mutation": "middle_byte_of_tensors.safetensors",
        "candidate_changes_authorized": False,
        "gate_changes_authorized": False,
        "full_replay_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 hostile-repair config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
