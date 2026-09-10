"""Freeze the R23 live verifier around the immutable positive candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

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
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base = "results/semantic_replication_r23/hidden_v1_result"
    source = "results/semantic_replication_r23/hidden_v1_source"
    files = {
        "r23_config": "experiments/semantic_replication_r23/configs/hidden_v1.json",
        "seed_reveal": (
            "experiments/semantic_replication_r23/configs/hidden_v1_reveal.json"
        ),
        "source_receipt": f"{source}/receipt.json",
        "source_rows": f"{source}/source_observations.jsonl",
        "candidate_wrapper": f"{base}/result.json",
        "candidate_engine": f"{base}/engine/result.json",
        "candidate_observations": f"{base}/engine/observations.jsonl",
        "candidate_cpu_observations": f"{base}/engine/cpu_observations.jsonl",
        "stored_strict_verification": f"{base}/strict_verification.json",
    }
    config = {
        "format": "abi-r23-live-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in LIVE_CODE_PATHS
        },
        **{name: _binding(root, relative) for name, relative in files.items()},
        "full_gpu_replay_required": True,
        "cpu_task_cover_required": True,
        "lifecycle_replay_required": True,
        "targeted_corruption": "middle_byte_of_tensors.safetensors",
        "candidate_changes_authorized": False,
        "gate_changes_authorized": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R23 live config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
