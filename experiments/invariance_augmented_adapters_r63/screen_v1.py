"""Screen the single R63 successor on the disclosed R60 development matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from abi.capability_compiler_phase2_common import sha256_file
from abi.layercake_core_loader import (
    CAPABILITY_CAKE_ORDER,
    SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE,
)
from experiments.broad_payload_reconstruction_r51 import screen_v1a as common
from experiments.external_router_cake_r49 import run_v1 as r49
from experiments.external_router_replication_r50 import run_v1a as r50
from experiments.irreversible_collapse_invariant_r59 import screen_v1 as r59
from experiments.isolated_capability_cakes_r53 import fit_router


CANDIDATE_SHA256 = "83610a38bfe25ba5cc7b7d1970a749dd9199d4965478fc020627aa80851b87d4"
METADATA_SHA256 = "bac92bb817f62c3c7907c8dbe5c6778aaa7381afa701900c4bbd82a74ff3d0f2"
AUGMENTATION_RECEIPT_SHA256 = "5ca1dbf4c8436156a3a3199dd4187f6327083c6393617cbc176b614d7d4cb1d0"
ROUTER_SHA256 = "ba88cf2a18ada898d739aa14a25805dae1c5212ffdcb4bfbfbd99c1c43b682db"
CATALOG_SHA256 = "4b0087c9a7fa0e0fd6f607fdbd94fffbdd3cb375f5880a0588c97414583a2ad7"
SOURCE_RECEIPT_SHA256 = "fc2384e461d5a43ba05e5abe7bcdb47220ab1acec1de990fb100d36e39505f03"
SOURCE_OUTPUTS_SHA256 = "35a5a6e1035035db1663d793257274e3be2b766aab79a00a0476478d0428b596"
PARENT_COUNTS = {"final_test": 813}


def _preflight(candidate: Path) -> None:
    if (
        sha256_file(candidate / "model.safetensors") != CANDIDATE_SHA256
        or sha256_file(candidate / "metadata.json") != METADATA_SHA256
        or sha256_file(candidate / "augmentation_receipt.json")
        != AUGMENTATION_RECEIPT_SHA256
    ):
        raise common.ScreenError("R63 candidate identity changed")
    metadata = json.loads((candidate / "metadata.json").read_text(encoding="utf-8"))
    acquired = metadata.get("acquired_core", {})
    architecture = metadata.get("architecture", {})
    expansion = acquired.get("capability_cake_expansion", {})
    training = metadata.get("training", {})
    receipt = json.loads(
        (candidate / "augmentation_receipt.json").read_text(encoding="utf-8")
    )
    if (
        architecture.get("architecture_version")
        != SIX_BLOCK_DEEP_CAPABILITY_ADAPTER_ARCHITECTURE
        or architecture.get("layers") != 6
        or architecture.get("task_cakes") != 14
        or tuple(architecture.get("capability_cake_order", ())) != CAPABILITY_CAKE_ORDER
        or architecture.get("capability_adapter_rank") != 32
        or architecture.get("capability_adapter_shared_across_layers") is not False
        or acquired.get("trainable_scope") != "deep_capability_adapter_cakes"
        or acquired.get("trainable_parameter_count") != 5_787_086
        or acquired.get("frozen_parameter_count") != 81_923_342
        or acquired.get("frozen_shared_state_preserved_exact") is not True
        or acquired.get("frozen_shared_state_sha256_before")
        != acquired.get("frozen_shared_state_sha256_after")
        or acquired.get("maximum_active_task_cakes_per_sequence") != 1
        or expansion.get("installed_capability_cakes") != 14
        or expansion.get("installed_deep_adapters") != 84
        or expansion.get("active_adapter_parameters") != 304_128
        or expansion.get("maximum_active_deep_adapters_per_sequence") != 6
        or expansion.get("extra_kv_positions") != 0
        or training.get("successful_optimizer_steps") != 6_000
        or training.get("seed") != 63_001
        or training.get("parent_logit_preservation_weight") != 0.5
        or receipt.get("augmented_records") != 24_421
        or receipt.get("new_teacher_outputs") != 0
        or receipt.get("r60_outputs_used_for_training") != 0
        or receipt.get("teacher_calls") != 0
        or receipt.get("checkpoint_sha256") != CANDIDATE_SHA256
        or receipt.get("metadata_sha256") != METADATA_SHA256
    ):
        raise common.ScreenError("R63 training or augmentation contract changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--router", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--broad-bundle", type=Path, required=True)
    parser.add_argument("--anchor-bundle", type=Path, required=True)
    parser.add_argument("--layercake-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = args.candidate.resolve()
    _preflight(candidate)
    if sha256_file(args.catalog.resolve()) != CATALOG_SHA256:
        raise common.ScreenError("R63 development catalog changed")
    if (
        sha256_file(args.source_dir.resolve() / "receipt.json") != SOURCE_RECEIPT_SHA256
        or sha256_file(args.source_dir.resolve() / "source_outputs.jsonl")
        != SOURCE_OUTPUTS_SHA256
    ):
        raise common.ScreenError("R63 development source changed")

    # Isolated process-local bindings let the frozen generic screen validate
    # the post-R59 catalog/source lineage without changing historical code.
    r49.CATALOG_SHA256 = CATALOG_SHA256
    r50.SOURCE_RECEIPT_SHA256 = SOURCE_RECEIPT_SHA256
    r50.SOURCE_OUTPUTS_SHA256 = SOURCE_OUTPUTS_SHA256
    telemetry: dict[str, Any] = {
        "candidate_transitions_checked": 0,
        "accepted_tokens": 0,
        "rejected_boundary_tokens": 0,
        "rejections_by_condition": {
            "maximum_identical_token_run": 0,
            "repeated_novel_lexical_fourgrams": 0,
        },
        "language_model_eos_stops": 0,
        "inherited_r55_lexical_truncations": 0,
    }

    def generate(model, tokenizer, prompt, route, maximum, device):
        return r59._generate(
            model,
            tokenizer,
            prompt,
            route,
            maximum,
            device,
            telemetry=telemetry,
        )

    common.run(
        candidate,
        args.router.resolve(),
        args.catalog.resolve(),
        "final_test",
        [],
        args.source_dir.resolve(),
        args.broad_bundle.resolve(),
        args.anchor_bundle.resolve(),
        args.layercake_root.resolve(),
        args.output.resolve(),
        candidate_sha256=CANDIDATE_SHA256,
        metadata_sha256=METADATA_SHA256,
        parent_counts=PARENT_COUNTS,
        result_format="abi-r63-invariance-augmented-development-screen/1",
        campaign_name="R63",
        expected_training_seed=63_001,
        router_sha256=ROUTER_SHA256,
        capability_to_route=fit_router.CAPABILITY_TO_INDEX,
        generation_fn=generate,
        extra_result_fields={
            "augmentation_receipt_sha256": AUGMENTATION_RECEIPT_SHA256,
            "r60_baseline_functional": 813,
            "runtime_telemetry": telemetry,
        },
    )


if __name__ == "__main__":
    main()

