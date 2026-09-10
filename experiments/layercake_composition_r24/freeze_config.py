"""Freeze R24 before domain-package training."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

from .binding import CODE_PATHS, GATES
from .protocol import NAMESPACES, SEEDS


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
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    files = {
        "r16_source_rows": "results/factual_semantic_r16/heldout_v2/source_observations.jsonl",
        "r16_receipt": "results/factual_semantic_r16/heldout_v2/receipt.json",
        "r16_strict_verification": "results/factual_semantic_r16/heldout_v2_strict_v4.json",
        "r23_config": "experiments/semantic_replication_r23/configs/hidden_v1.json",
        "r23_reveal": (
            "experiments/semantic_replication_r23/configs/hidden_v1_reveal.json"
        ),
        "r23_engine": "results/semantic_replication_r23/hidden_v1_result/engine/result.json",
        "r23_live_verification": (
            "results/semantic_replication_r23/hidden_v2_live/strict_verification.json"
        ),
        "r23_hidden_rows": (
            "results/semantic_replication_r23/hidden_v1_result/engine/observations.jsonl"
        ),
    }
    engine = json_object(
        root / "results/generative_transfer_r21/public_v5_bakeoff/engine/result.json"
    )
    english = sorted(
        (
            {
                "path": package["path"],
                "bytes": (root / package["path"]).stat().st_size,
                "sha256": sha256_file(root / package["path"]),
                "cake_id": package["cake_id"],
            }
            for package in engine["systems"]["21022"]["abi_factorized"]["packages"]
        ),
        key=lambda row: row["cake_id"],
    )
    config = {
        "format": "abi-r24-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        **{name: _binding(root, relative) for name, relative in files.items()},
        "english_packages": english,
        "namespaces": list(NAMESPACES),
        "seeds": list(SEEDS),
        "training": {
            "device": "cuda",
            "steps": 2000,
            "batch_size": 8,
            "learning_rate": 0.0008,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "row_exposure_per_package": 16000,
        },
        "model": {
            "model_width": 48,
            "attention_heads": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "feedforward_width": 128,
            "pointer_width": 24,
            "dropout": 0.0,
            "maximum_source_lexemes": 128,
            "maximum_target_actions": 32,
        },
        "gates": GATES,
        "new_source_calls_authorized": False,
        "english_package_changes_authorized": False,
        "evaluation_training_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R24 config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
