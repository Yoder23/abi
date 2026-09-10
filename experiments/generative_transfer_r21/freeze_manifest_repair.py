"""Preregister the additive R21 LayerCake manifest repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .manifest_repair_binding import MANIFEST_REPAIR_CODE_PATHS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    failed_root = root / "results/generative_transfer_r21/public_v4_bakeoff_run2"
    packages = sorted((failed_root / "engine/packages").glob("*.cake"))
    if len(packages) != 24:
        raise RuntimeError("R21 failed package inventory is not 24")
    assurance = "experiments/generative_transfer_r21/configs/hash_assurance_v4.json"
    failure = "results/generative_transfer_r21/public_v4_bakeoff_run2/operational_failure.json"
    config = {
        "format": "abi-r21-manifest-repair-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in MANIFEST_REPAIR_CODE_PATHS
        },
        "hash_assurance_config": {
            "path": assurance,
            "sha256": sha256_file(root / assurance),
        },
        "failed_attempt_record": {
            "path": failure,
            "sha256": sha256_file(root / failure),
        },
        "failed_package_inventory": [
            {
                "path": package.relative_to(root).as_posix(),
                "bytes": package.stat().st_size,
                "sha256": sha256_file(package),
            }
            for package in packages
        ],
        "input_contract_addition": {"mode": "direct_selected_portable_decoder"},
        "retraining_authorized": True,
        "data_changes_authorized": False,
        "model_changes_authorized": False,
        "gate_changes_authorized": False,
        "layercake_changes_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 manifest-repair config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
