"""Preregister fresh live verification of the exact R21 public candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

from .live_binding import LIVE_CODE_PATHS


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
    result_root = "results/generative_transfer_r21/public_v5_bakeoff"
    engine = json_object(root / f"{result_root}/engine/result.json")
    packages = sorted(
        (
            {
                "path": package["path"],
                "bytes": (root / package["path"]).stat().st_size,
                "sha256": sha256_file(root / package["path"]),
            }
            for seed in engine["systems"].values()
            for method in seed.values()
            for package in method["packages"]
        ),
        key=lambda row: row["path"],
    )
    if len(packages) != 24:
        raise RuntimeError("R21 candidate package inventory is not 24")
    layercake_root = (root / "../layercake_release/layercake").resolve()
    layercake = []
    for target in sorted(layercake_root.rglob("*.py")):
        relative = Path(os.path.relpath(target, root)).as_posix()
        layercake.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    files = {
        "manifest_repair_config": "experiments/generative_transfer_r21/configs/manifest_repair_v5.json",
        "candidate_wrapper": f"{result_root}/result.json",
        "candidate_engine": f"{result_root}/engine/result.json",
        "candidate_observations": f"{result_root}/engine/observations.jsonl",
        "candidate_labeler": f"{result_root}/engine/labeler.json",
        "assurance_binding": f"{result_root}/assurance_binding.json",
        "manifest_repair_binding": f"{result_root}/manifest_repair_binding.json",
    }
    config = {
        "format": "abi-r21-live-verification-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in LIVE_CODE_PATHS},
        **{name: _binding(root, relative) for name, relative in files.items()},
        "packages": packages,
        "layercake_python": layercake,
        "layercake_python_file_count": len(layercake),
        "candidate_verdict": "PASS_PUBLIC_PREREQUISITE",
        "required": {
            "gpu_rows": 1080,
            "cpu_rows": 54,
            "package_removals": 24,
            "package_restorations": 24,
            "corrupt_packages_rejected": 24,
            "signed_packages": 24,
            "source_software_loaded": False,
        },
        "stored_scientific_booleans_trusted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 live-verification config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
