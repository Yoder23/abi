"""Recompute explicit imported-information accounting for R15B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .protocol import heldout_capabilities
from .public_extraction import _anchor_rows


def account(config_path: Path, reveal_path: Path, run_dir: Path) -> dict[str, Any]:
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    receipt = json_object(run_dir / "receipt.json")
    heldout = heldout_capabilities(
        str(reveal["secret_hex"]),
        expected_commitment=str(config["heldout_seed_commitment"]),
        count=int(config["data"]["heldout_capabilities"]),
    )
    prompts = [row["prompt"] for item in heldout for row in _anchor_rows(item.slot_order)]
    unique_prompts = sorted(set(prompts))
    source_rows = [
        json.loads(line)
        for line in (run_dir / receipt["source"]["observations"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(source_rows) != len(prompts):
        raise R14Error("R15B accounting source rows changed")
    bundles = receipt["source"]["capability_receipts"]
    package_items = receipt["packages"]["after"]
    label_items = receipt["semantic_labels"]
    residual_values = sum(
        int(item["bundle"]["residual_shape"][0])
        * int(item["bundle"]["residual_shape"][1])
        * int(item["bundle"]["residual_shape"][2])
        for item in bundles
    )
    output_row_values = sum(
        int(item["bundle"]["output_rows_shape"][0]) * int(item["bundle"]["output_rows_shape"][1])
        for item in bundles
    )
    result = {
        "format": "abi-r15b-imported-information-accounting/1",
        "raw_source_prompts": len(prompts),
        "unique_source_prompts": len(unique_prompts),
        "unique_prompt_utf8_bytes": sum(len(value.encode()) for value in unique_prompts),
        "teacher_generated_output_bytes": sum(
            len(str(row["completion"]).encode()) for row in source_rows
        ),
        "teacher_generated_tokens": sum(
            int(item["metrics"]["generated_tokens"]) for item in bundles
        ),
        "canonical_logits_stored": len(source_rows) * 8,
        "hidden_activation_values_stored": residual_values,
        "hidden_activation_bytes": residual_values * 4,
        "source_output_weight_values_stored": output_row_values,
        "source_output_weight_bytes": output_row_values * 4,
        "unique_source_output_parameters_copied_transiently": 8
        * int(bundles[0]["bundle"]["output_rows_shape"][1]),
        "frozen_source_parameters_in_final_packages": 0,
        "final_transition_parameters": len(package_items) * 3 * 8 * 8,
        "bridge_parameters_trained": 0,
        "source_training_steps": 0,
        "recipient_training_steps": 0,
        "representation_bundle_disk_bytes": sum(int(item["bundle"]["bytes"]) for item in bundles),
        "final_package_disk_bytes": sum(int(item["bytes"]) for item in package_items),
        "semantic_label_disk_bytes": sum(
            (run_dir / str(item["path"])).stat().st_size for item in label_items
        ),
        "source_model_inference_seconds": sum(
            float(item["metrics"]["wall_seconds"]) for item in bundles
        ),
        "external_hardware_used": False,
        "peak_gpu_memory_bytes": "NOT_MEASURED",
        "peak_cpu_ram_bytes": "NOT_MEASURED",
        "claim_ceiling": "ACCOUNTING_FOR_R15B_ONLY_NOT_GLOBAL_MINIMALITY",
        "source_rows_sha256": sha256_file(run_dir / receipt["source"]["observations"]["path"]),
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = account(args.config, args.reveal, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
