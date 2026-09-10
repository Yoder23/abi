"""Fresh physical extraction and runtime replay for the frozen R18 mechanism."""

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


def run_live(source_run: Path, original: Path, output: Path) -> dict[str, Any]:
    original_strict = verify(source_run, original)
    live_receipt = run(source_run, output)
    live_strict = verify(source_run, output)
    original_receipt = json_object(original / "receipt.json")
    exact_files = (
        "source_bundle.json",
        "mood_permutation_bundle.json",
        "evaluation.jsonl",
    )
    comparisons = []
    for relative in exact_files:
        before = original / relative
        after = output / relative
        if not before.is_file() or not after.is_file() or before.read_bytes() != after.read_bytes():
            raise R14Error(f"R18 live evidence changed: {relative}")
        comparisons.append({"path": relative, "sha256": sha256_file(after)})
    for key in ("package", "control_package"):
        relative = Path(str(original_receipt[key]["path"]))
        before = original / relative
        after = output / relative
        if not before.is_file() or not after.is_file() or before.read_bytes() != after.read_bytes():
            raise R14Error(f"R18 live {key} changed")
        comparisons.append({"path": str(relative), "sha256": sha256_file(after)})
    if (
        original_strict["claim"] != live_strict["claim"]
        or original_receipt["metrics"] != live_receipt["metrics"]
        or live_receipt["verdict"] != "PASS"
        or live_strict["verdict"] != "PASS"
    ):
        raise R14Error("R18 live scientific result changed")
    result = {
        "format": "abi-r18-live-verification/1",
        "verdict": "PASS",
        "files_replayed_byte_exact": comparisons,
        "source_rows_reverified": original_strict["source_rows_verified"],
        "evaluation_rows_replayed": original_strict["evaluation_rows_verified"],
        "packages_replayed": 2,
        "physical_extractions_replayed": 2,
        "teacher_present_at_compilation": False,
        "teacher_present_at_package_execution": False,
        "source_training_steps": 0,
        "host_training_steps": 0,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "live_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_live(args.source_run, args.original, args.output), indent=2))


if __name__ == "__main__":
    main()
