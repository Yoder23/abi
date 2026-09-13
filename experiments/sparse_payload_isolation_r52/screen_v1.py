"""Bind the disclosed-screen engine to the exact R52 sparse endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.broad_payload_reconstruction_r51 import screen_v1a as common


CANDIDATE_SHA256 = "5ef352f6638f17d3df96e8dae3be160648726f371caf54d29afa1ec835edd58b"
METADATA_SHA256 = "b18882aaba781e3bef0f5b7a29d6b1e0668c7c9fdf900f34b30fcf7e39687e4c"
PARENT_COUNTS = {"validation": 1_277, "final_test": 1_261}


def _preflight(candidate: Path) -> None:
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    acquired = metadata.get("acquired_core", {})
    training = metadata.get("training", {})
    if (
        acquired.get("trainable_scope") != "task_cakes_classifier"
        or acquired.get("trainable_parameter_count") != 1_006_090
        or acquired.get("frozen_parameter_count") != 81_912_576
        or acquired.get("frozen_shared_state_preserved_exact") is not True
        or acquired.get("frozen_shared_state_sha256_before")
        != acquired.get("frozen_shared_state_sha256_after")
        or acquired.get("graph_topology_changed") is not False
        or acquired.get("maximum_active_task_cakes_per_sequence") != 1
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("seed") != 52_001
    ):
        raise common.ScreenError("R52 sparse training contract changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(PARENT_COUNTS), required=True)
    parser.add_argument("--source-bundle", type=Path, action="append", default=[])
    parser.add_argument("--live-source-dir", type=Path)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    _preflight(candidate)
    common.run(
        candidate, args.router.resolve(), args.catalog.resolve(), args.split,
        [path.resolve() for path in args.source_bundle],
        args.live_source_dir.resolve() if args.live_source_dir else None,
        args.broad_bundle.resolve(), args.anchor_bundle.resolve(),
        args.layercake_root.resolve(), args.output.resolve(),
        candidate_sha256=CANDIDATE_SHA256,
        metadata_sha256=METADATA_SHA256,
        parent_counts=PARENT_COUNTS,
        result_format="abi-r52-sparse-payload-development-screen/1",
        campaign_name="R52",
        expected_training_seed=52_001,
    )


if __name__ == "__main__":
    main()
