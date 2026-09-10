"""Freeze additive R24 negative verification before live replay."""

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
    run_root = Path("results/layercake_composition_r24/public_v1")
    result = json_object(root / run_root / "result.json")
    inventory = [
        run_root / "result.json",
        run_root / "domain_observations.jsonl",
        run_root / "english_immutability.jsonl",
        run_root / "lifecycle.jsonl",
        run_root / "english_leakage.jsonl",
    ]
    inventory.extend(
        Path(system["package"]["path"])
        for seed in result["systems"].values()
        for system in seed.values()
    )
    relative_inventory = sorted(path.as_posix() for path in inventory)
    if len(relative_inventory) != 11 or len(set(relative_inventory)) != 11:
        raise RuntimeError("R24 negative-verification inventory changed")
    config = {
        "format": "abi-r24-negative-verification-config/1",
        "implementation_freeze_commit": commit,
        "code_sha256": {
            relative: sha256_file(root / relative) for relative in CODE_PATHS
        },
        "original_config": [
            _binding(
                root,
                "experiments/layercake_composition_r24/configs/public_v1.json",
            )
        ],
        "result_inventory": [
            _binding(root, relative) for relative in relative_inventory
        ],
        "candidate_changes": 0,
        "gate_changes": 0,
        "scorer_changes": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError("refusing to overwrite R24 negative-verification config")
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
