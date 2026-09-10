"""Freeze the public R21 implementation and scientific contract."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .binding import CODE_PATHS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    config = {
        "format": "abi-r21-public-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {relative: sha256_file(root / relative) for relative in CODE_PATHS},
        "source": {
            "model_id": "Qwen/Qwen2-7B-Instruct",
            "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
            "device": "cuda",
            "max_new_tokens": 96,
            "max_label_tokens": 4,
        },
        "data": {"training_rows": 600, "evaluation_rows": 120, "tasks": 6},
        "methods": ["raw_sequence", "labeled_monolith", "abi_factorized"],
        "seeds": [21021, 21022, 21023],
        "model": {
            "monolith": {"model_width": 96, "attention_heads": 4, "encoder_layers": 2, "decoder_layers": 2, "feedforward_width": 256, "pointer_width": 48},
            "factor": {"model_width": 48, "attention_heads": 4, "encoder_layers": 1, "decoder_layers": 1, "feedforward_width": 128, "pointer_width": 24},
            "dropout": 0.0,
            "maximum_source_lexemes": 128,
            "maximum_target_actions": 256,
        },
        "training": {
            "learning_rate": 0.0008,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "monolith_steps": 2500,
            "monolith_batch_size": 24,
            "factor_steps_per_task": 1000,
            "factor_batch_size": 10,
            "row_exposures_per_method": 60000,
            "device": "cuda",
            "precision": "fp32",
        },
        "gates": {
            "minimum_label_exact": 114,
            "minimum_task_functional": 15,
            "minimum_non_hallucinating": 114,
            "minimum_non_collapsed": 114,
            "maximum_factor_to_labeled_parameters": 1.25,
            "require_all_three_seeds": True,
            "require_no_worse_than_teacher_and_controls": True,
            "bootstrap_samples": 10000,
        },
        "layercake": {
            "root": "../layercake_release",
            "abi_version": "lc-direct-neural-decoder/1",
            "abi_sha256": "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a",
            "research_signing_seed_hex": "d4603cc928b438a43f40594f39effcd600e9b783b442ac298981cbbe258d15cd",
        },
        "evaluation_teacher": {
            "path": "results/instructional_realization_r20/public_v1_source/source_observations.jsonl",
            "sha256": sha256_file(
                root / "results/instructional_realization_r20/public_v1_source/source_observations.jsonl"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R21 config")
    args.output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
