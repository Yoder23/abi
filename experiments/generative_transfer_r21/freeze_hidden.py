"""Preregister the R21 hidden replication before seed reveal."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

from .hidden_binding import HIDDEN_CODE_PATHS


def _binding(root: Path, relative: str) -> dict[str, object]:
    target = root / relative
    return {
        "path": relative,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-commitment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.seed_commitment):
        raise RuntimeError("R21 hidden seed commitment is invalid")
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
    files = {
        "base_config": "experiments/generative_transfer_r21/configs/public_v1.json",
        "public_candidate_wrapper": f"{result_root}/result.json",
        "public_candidate_engine": f"{result_root}/engine/result.json",
        "public_candidate_labeler": f"{result_root}/engine/labeler.json",
        "public_live_receipt": "results/generative_transfer_r21/public_v7_live_verification/receipt.json",
        "public_strict_verification": "results/generative_transfer_r21/public_v7_live_verification/strict_verification.json",
    }
    config = {
        "format": "abi-r21-hidden-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in HIDDEN_CODE_PATHS},
        **{name: _binding(root, relative) for name, relative in files.items()},
        "packages": packages,
        "seed_commitment": args.seed_commitment,
        "source": {
            "model_id": "Qwen/Qwen2-7B-Instruct",
            "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "max_new_tokens": 96,
            "device": "cuda",
        },
        "rows": {"total": 120, "per_task": 20, "tasks": 6},
        "methods": ["raw_sequence", "labeled_monolith", "abi_factorized"],
        "seeds": [21021, 21022, 21023],
        "gates": {
            "minimum_label_exact": 114,
            "minimum_task_functional_per_seed": 15,
            "minimum_non_hallucinating_per_seed": 114,
            "minimum_non_collapsed_per_seed": 114,
            "require_each_seed_no_worse_than_teacher": True,
            "require_each_seed_no_worse_than_controls": True,
            "bootstrap_samples": 10000,
        },
        "student_retraining_authorized": False,
        "second_reveal_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 hidden config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
