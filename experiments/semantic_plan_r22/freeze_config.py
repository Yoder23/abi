"""Freeze R22 before source normalization."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .binding import CODE_PATHS


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
    files = {
        "base_training_config": "experiments/generative_transfer_r21/configs/public_v1.json",
        "raw_source_receipt": "results/generative_transfer_r21/public_v3_source/receipt.json",
        "raw_source_rows": "results/generative_transfer_r21/public_v3_source/source_observations.jsonl",
        "r21_hidden_result": "results/generative_transfer_r21/hidden_v1_result/result.json",
        "r21_hidden_verification": "results/generative_transfer_r21/hidden_v1_result/strict_verification.json",
    }
    config = {
        "format": "abi-r22-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        **{name: _binding(root, relative) for name, relative in files.items()},
        "source": {
            "model_id": "Qwen/Qwen2-7B-Instruct",
            "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "device": "cuda",
            "max_new_tokens": 128,
        },
        "rows": 600,
        "normalization_calls": 600,
        "gates": {
            "minimum_functional": 594,
            "minimum_non_hallucinating": 600,
            "minimum_non_collapsed": 594,
            "required_fields_verbatim_once": 600,
            "required_label_exact": 600,
        },
        "retry_authorized": False,
        "evaluation_access_authorized": False,
        "student_training_before_source_pass": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R22 config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
