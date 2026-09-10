"""Freeze R23 before revealing its hidden row seed."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

from .binding import CODE_PATHS, GATES, _jsonl
from .protocol import validate_public_lexicon


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
        raise RuntimeError("R23 seed commitment is invalid")
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
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
        "public_candidate_engine": f"{result_root}/engine/result.json",
        "public_candidate_labeler": f"{result_root}/engine/labeler.json",
        "public_live_receipt": (
            "results/generative_transfer_r21/public_v7_live_verification/receipt.json"
        ),
        "public_strict_verification": (
            "results/generative_transfer_r21/public_v7_live_verification/"
            "strict_verification.json"
        ),
        "r21_hidden_result": "results/generative_transfer_r21/hidden_v1_result/result.json",
        "r21_hidden_verification": (
            "results/generative_transfer_r21/hidden_v1_result/strict_verification.json"
        ),
        "r22_failed_receipt": "results/semantic_plan_r22/public_v1_source/receipt.json",
        "public_semantic_corpus": (
            "results/generative_transfer_r21/public_v3_source/source_observations.jsonl"
        ),
    }
    public_rows = _jsonl(root / files["public_semantic_corpus"])
    config = {
        "format": "abi-r23-hidden-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        **{name: _binding(root, relative) for name, relative in files.items()},
        "packages": packages,
        "public_marker_counts": validate_public_lexicon(public_rows),
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
        "gates": GATES,
        "student_retraining_authorized": False,
        "package_changes_authorized": False,
        "second_reveal_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R23 config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
