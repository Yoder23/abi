"""Preregister the R21 registered-ontology label control."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .label_control_binding import CONTROL_CODE_PATHS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    bindings = {
        "base_config": "experiments/generative_transfer_r21/configs/public_v1.json",
        "failed_source_receipt": "results/generative_transfer_r21/public_v1_source/receipt.json",
        "failed_source_rows": "results/generative_transfer_r21/public_v1_source/source_observations.jsonl",
    }
    config = {
        "format": "abi-r21-label-control-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in CONTROL_CODE_PATHS},
        **{
            name: {"path": relative, "sha256": sha256_file(root / relative)}
            for name, relative in bindings.items()
        },
        "required_training_exact": 600,
        "unique_registered_seed_pairs": 36,
        "teacher_side_labeling": False,
        "response_regeneration_authorized": False,
        "evaluation_access_authorized": False,
        "autonomous_ontology_claim_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 label-control config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
