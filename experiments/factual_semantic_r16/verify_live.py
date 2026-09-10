"""Fresh full source, physical extraction, and runtime replay for R16."""

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

from .run import run
from .verify import verify


def run_live(
    config: Path, reveal: Path, original: Path, output: Path
) -> dict[str, Any]:
    original_strict = verify(config, reveal, original)
    live_receipt = run(config, reveal, output)
    live_strict = verify(config, reveal, output)
    exact_files = (
        "source_observations.jsonl",
        "source_prompt_end_residuals.safetensors",
        "source_bundle.json",
        "rotated_score_bundle.json",
        "evaluation.jsonl",
    )
    comparisons = []
    for relative in exact_files:
        left = original / relative
        right = output / relative
        if not left.is_file() or not right.is_file() or left.read_bytes() != right.read_bytes():
            raise R14Error(f"R16 live evidence changed: {relative}")
        comparisons.append({"path": relative, "sha256": sha256_file(right)})
    original_run = json_object(original / "receipt.json")
    for item in original_run["packages"]:
        relative = Path("extraction") / item["path"]
        left = original / relative
        right = output / relative
        if left.read_bytes() != right.read_bytes():
            raise R14Error(f"R16 live package changed: {item['namespace']}")
        comparisons.append({"path": str(relative), "sha256": sha256_file(right)})
    if (
        original_strict["claim"] != live_strict["claim"]
        or original_run["metrics"] != live_receipt["metrics"]
        or live_receipt["verdict"] != "PASS"
    ):
        raise R14Error("R16 live scientific receipt changed")
    result = {
        "format": "abi-r16-live-verification/1",
        "verdict": "PASS",
        "files_replayed_byte_exact": comparisons,
        "source_rows_replayed": original_run["information_accounting"]["raw_source_prompts"],
        "evaluation_rows_replayed": original_run["metrics"]["evaluation_rows"],
        "packages_replayed": len(original_run["packages"]),
        "physical_extractions_replayed": 2,
        "source_training_steps": 0,
        "teacher_present_at_package_execution": False,
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "live_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_live(args.config, args.reveal, args.original, args.output), indent=2))


if __name__ == "__main__":
    main()
