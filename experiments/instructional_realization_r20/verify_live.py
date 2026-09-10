"""Fresh physical replay of the R20 public compiler and runtime."""

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

from .run_public import run
from .verify import verify


def run_live(config_path: Path, source_run: Path, original: Path, output: Path) -> dict[str, Any]:
    original_strict = verify(config_path, source_run, original)
    live_receipt = run(config_path, source_run, output)
    live_strict = verify(config_path, source_run, output)
    original_receipt = json_object(original / "receipt.json")
    comparisons = []
    for relative in (
        "source_bundle.json",
        "rotated_instruction_bundle.json",
        "evaluation.jsonl",
    ):
        before = original / relative
        after = output / relative
        if before.read_bytes() != after.read_bytes():
            raise R14Error(f"R20 live evidence changed: {relative}")
        comparisons.append({"path": relative, "sha256": sha256_file(after)})
    for name in ("package", "control_package"):
        relative = Path(original_receipt[name]["path"])
        before = original / relative
        after = output / relative
        if before.read_bytes() != after.read_bytes():
            raise R14Error(f"R20 live {name} changed")
        comparisons.append({"path": relative.as_posix(), "sha256": sha256_file(after)})
    if (
        original_receipt["metrics"] != live_receipt["metrics"]
        or original_strict["verdict"] != "PASS"
        or live_strict["verdict"] != "PASS"
    ):
        raise R14Error("R20 live scientific result changed")
    result = {
        "format": "abi-r20-public-live-verification/1",
        "verdict": "PASS",
        "files_replayed_byte_exact": comparisons,
        "source_rows_reverified": 192,
        "evaluation_rows_replayed": 120,
        "physical_primary_extractions_replayed": 1,
        "physical_control_extractions_replayed": 1,
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
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_live(args.config, args.source_run, args.original, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
