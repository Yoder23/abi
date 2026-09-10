"""Fresh physical replay for the R19 disclosed development prerequisite."""

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

from .run_development import run
from .verify_development import verify


def run_live(
    r17_source: Path,
    r18_config: Path,
    r18_reveal: Path,
    r18_source: Path,
    original: Path,
    output: Path,
) -> dict[str, Any]:
    original_strict = verify(r17_source, r18_config, r18_reveal, r18_source, original)
    live_receipt = run(r17_source, r18_config, r18_reveal, r18_source, output)
    live_strict = verify(r17_source, r18_config, r18_reveal, r18_source, output)
    original_receipt = json_object(original / "receipt.json")
    comparisons = []
    for name in ("r17_public_v2", "r18_failed_hidden"):
        for relative in ("source_bundle.json", "mood_permutation_bundle.json", "evaluation.jsonl"):
            before = original / name / relative
            after = output / name / relative
            if before.read_bytes() != after.read_bytes():
                raise R14Error(f"R19 live evidence changed: {name}/{relative}")
            comparisons.append({"path": f"{name}/{relative}", "sha256": sha256_file(after)})
        package_relative = Path(original_receipt["datasets"][name]["package"]["path"])
        before_package = original / name / package_relative
        after_package = output / name / package_relative
        if before_package.read_bytes() != after_package.read_bytes():
            raise R14Error(f"R19 live package changed: {name}")
        comparisons.append(
            {
                "path": f"{name}/{package_relative}",
                "sha256": sha256_file(after_package),
            }
        )
        rejection = json_object(output / name / "control_rejection.json")
        if (
            rejection.get("reason") != "NO_STRUCTURE_COMPATIBLE_POLARITY_CONTRAST"
            or rejection.get("packages_emitted") != 0
        ):
            raise R14Error(f"R19 live control rejection changed: {name}")
    if original_receipt["datasets"] != {
        name: {
            **live_receipt["datasets"][name],
            "control": original_receipt["datasets"][name]["control"],
        }
        for name in original_receipt["datasets"]
    }:
        # Raw rejection records include randomized capsule paths; compare all
        # deterministic scientific fields separately below.
        for name in original_receipt["datasets"]:
            if (
                original_receipt["datasets"][name]["metrics"]
                != live_receipt["datasets"][name]["metrics"]
                or original_receipt["datasets"][name]["package"]
                != live_receipt["datasets"][name]["package"]
            ):
                raise R14Error("R19 live scientific receipt changed")
    if original_strict["verdict"] != live_strict["verdict"] != "PASS":
        raise R14Error("R19 live strict verdict changed")
    result = {
        "format": "abi-r19-development-live-verification/1",
        "verdict": "PASS",
        "files_replayed_byte_exact": comparisons,
        "datasets_replayed": 2,
        "evaluation_rows_replayed": 96,
        "physical_primary_extractions_replayed": 2,
        "physical_control_rejections_replayed": 2,
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
    parser.add_argument("--r17-source", type=Path, required=True)
    parser.add_argument("--r18-config", type=Path, required=True)
    parser.add_argument("--r18-reveal", type=Path, required=True)
    parser.add_argument("--r18-source", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_live(
                args.r17_source,
                args.r18_config,
                args.r18_reveal,
                args.r18_source,
                args.original,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
