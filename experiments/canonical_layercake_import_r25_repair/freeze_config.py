"""Freeze R25 scorer repair before the fresh replay."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments.foreign_capability_r14.core import json_object, sha256_file

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
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    run_root = Path("results/canonical_layercake_import_r25/public_v1")
    result = json_object(root / run_root / "result.json")
    inventory = [
        run_root / "result.json",
        *(run_root / ref["path"] for ref in result["artifacts"].values()),
        *(
            Path(package["path"])
            for build in result["builds"].values()
            for package in build.values()
        ),
    ]
    relative_inventory = sorted(path.as_posix() for path in inventory)
    if len(relative_inventory) != 12 or len(set(relative_inventory)) != 12:
        raise RuntimeError("R25 repair result inventory changed")
    config = {
        "format": "abi-r25-semantic-repair-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        "original_config": [
            _binding(root, "experiments/canonical_layercake_import_r25/configs/public_v1.json")
        ],
        "result_inventory": [
            _binding(root, relative) for relative in relative_inventory
        ],
        "candidate_changes": 0,
        "gate_changes": 0,
        "scorer_changes": 1,
        "scorer_change": "teacher agreement uses the pre-existing R16 normalized_text",
        "fresh_replication_required": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R25 repair config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
