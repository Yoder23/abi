"""Freeze the one R23 live metadata-source repair."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import sha256_file

from .live_repair_binding import REPAIR_CODE_PATHS


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
        "live_config": "experiments/semantic_replication_r23/configs/live_v1.json",
        "failed_attempt": "results/semantic_replication_r23/live_v1_failed_attempt.json",
        "r23_config": "experiments/semantic_replication_r23/configs/hidden_v1.json",
    }
    config = {
        "format": "abi-r23-live-repair-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in REPAIR_CODE_PATHS
        },
        **{name: _binding(root, relative) for name, relative in files.items()},
        "systems_source": "frozen_public_candidate_engine",
        "candidate_changes_authorized": False,
        "package_changes_authorized": False,
        "scorer_changes_authorized": False,
        "gate_changes_authorized": False,
        "full_replay_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R23 live-repair config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
