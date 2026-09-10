"""Preregister the additive R21 self-hash assurance repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .hash_assurance_binding import ASSURANCE_CODE_PATHS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    bindings = {
        "label_control_config": "experiments/generative_transfer_r21/configs/label_control_v3.json",
        "source_receipt": "results/generative_transfer_r21/public_v3_source/receipt.json",
        "source_rows": "results/generative_transfer_r21/public_v3_source/source_observations.jsonl",
        "source_labeler": "results/generative_transfer_r21/public_v3_source/labeler.json",
    }
    config = {
        "format": "abi-r21-hash-assurance-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in ASSURANCE_CODE_PATHS
        },
        **{
            name: {"path": relative, "sha256": sha256_file(root / relative)}
            for name, relative in bindings.items()
        },
        "removed_fields": ["evidence_sha256"],
        "data_changes_authorized": False,
        "gate_changes_authorized": False,
        "positive_stored-evidence_promotion_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 hash-assurance config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
