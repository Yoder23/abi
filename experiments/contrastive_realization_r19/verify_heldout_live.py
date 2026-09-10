"""Fresh physical replay of the frozen R19 held-out compilation and runtime."""

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

from .run_heldout import run
from .verify_heldout import verify


def run_live(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    original: Path,
    output: Path,
) -> dict[str, Any]:
    original_strict = verify(config_path, reveal_path, source_run, original)
    live_receipt = run(config_path, reveal_path, source_run, output)
    live_strict = verify(config_path, reveal_path, source_run, output)
    original_receipt = json_object(original / "receipt.json")
    comparisons = []
    for relative in (
        "hidden/source_bundle.json",
        "hidden/mood_permutation_bundle.json",
        "hidden/evaluation.jsonl",
    ):
        before = original / relative
        after = output / relative
        if before.read_bytes() != after.read_bytes():
            raise R14Error(f"R19 held-out live evidence changed: {relative}")
        comparisons.append({"path": relative, "sha256": sha256_file(after)})
    package_relative = Path("hidden") / original_receipt["dataset"]["package"]["path"]
    before_package = original / package_relative
    after_package = output / package_relative
    if before_package.read_bytes() != after_package.read_bytes():
        raise R14Error("R19 held-out live package changed")
    comparisons.append({"path": package_relative.as_posix(), "sha256": sha256_file(after_package)})
    rejection = json_object(output / "hidden/control_rejection.json")
    if (
        rejection.get("reason") != "NO_STRUCTURE_COMPATIBLE_POLARITY_CONTRAST"
        or rejection.get("packages_emitted") != 0
        or (output / "hidden/control_extraction").exists()
    ):
        raise R14Error("R19 held-out live control rejection changed")
    if (
        original_receipt["dataset"]["metrics"] != live_receipt["dataset"]["metrics"]
        or original_receipt["dataset"]["package"] != live_receipt["dataset"]["package"]
        or original_strict["verdict"] != "PASS"
        or live_strict["verdict"] != "PASS"
    ):
        raise R14Error("R19 held-out live scientific result changed")
    result = {
        "format": "abi-r19-heldout-live-verification/1",
        "verdict": "PASS",
        "files_replayed_byte_exact": comparisons,
        "datasets_replayed": 1,
        "evaluation_rows_replayed": 48,
        "physical_primary_extractions_replayed": 1,
        "physical_control_rejections_replayed": 1,
        "source_training_steps": 0,
        "host_training_steps": 0,
        "teacher_present_at_package_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "live_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_live(
                args.config,
                args.reveal,
                args.source_run,
                args.original,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
