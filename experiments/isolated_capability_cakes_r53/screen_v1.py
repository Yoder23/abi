"""Bind the disclosed-screen engine to the exact R53 isolated-cake endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from abi.layercake_core_loader import (
    CAPABILITY_CAKE_ORDER,
    SIX_BLOCK_CAPABILITY_CAKE_ARCHITECTURE,
)
from experiments.broad_payload_reconstruction_r51 import screen_v1a as common
from experiments.isolated_capability_cakes_r53 import fit_router


CANDIDATE_SHA256 = "d8d220792b42373ebc2a076a39301d4185905905596b526f91d5441ca5a1868c"
METADATA_SHA256 = "488b735057043e26d3c0ff10d09ee5abc88fd82d885b904008cdeec5339f7fad"
ROUTER_SHA256 = "ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db"
PARENT_COUNTS = {"validation": 1_277, "final_test": 1_261}


def _preflight(candidate: Path) -> None:
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    acquired = metadata.get("acquired_core", {})
    architecture = metadata.get("architecture", {})
    expansion = acquired.get("capability_cake_expansion", {})
    training = metadata.get("training", {})
    if (
        architecture.get("architecture_version")
        != SIX_BLOCK_CAPABILITY_CAKE_ARCHITECTURE
        or architecture.get("layers") != 6
        or architecture.get("task_cakes") != 14
        or tuple(architecture.get("capability_cake_order", ()))
        != CAPABILITY_CAKE_ORDER
        or acquired.get("trainable_scope") != "capability_cakes_classifier"
        or acquired.get("trainable_parameter_count") != 1_408_526
        or acquired.get("frozen_parameter_count") != 81_912_576
        or acquired.get("frozen_shared_state_preserved_exact") is not True
        or acquired.get("frozen_shared_state_sha256_before")
        != acquired.get("frozen_shared_state_sha256_after")
        or acquired.get("maximum_active_task_cakes_per_sequence") != 1
        or expansion.get("parent_cake_values_copied_exactly") is not True
        or expansion.get("classifier_rows_copied_exactly") is not True
        or expansion.get("initial_selected_cake_function_parent_equivalent") is not True
        or expansion.get("installed_capability_cakes") != 14
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("seed") != 53_001
    ):
        raise common.ScreenError("R53 isolated-cake training contract changed")


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
        candidate,
        args.router.resolve(),
        args.catalog.resolve(),
        args.split,
        [path.resolve() for path in args.source_bundle],
        args.live_source_dir.resolve() if args.live_source_dir else None,
        args.broad_bundle.resolve(),
        args.anchor_bundle.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
        candidate_sha256=CANDIDATE_SHA256,
        metadata_sha256=METADATA_SHA256,
        parent_counts=PARENT_COUNTS,
        result_format="abi-r53-isolated-capability-cakes-development-screen/1",
        campaign_name="R53",
        expected_training_seed=53_001,
        router_sha256=ROUTER_SHA256,
        capability_to_route=fit_router.CAPABILITY_TO_INDEX,
    )


if __name__ == "__main__":
    main()
