"""Run the frozen R21 bakeoff with the registered-ontology label control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    sha256_file,
    write_json_once,
)

from . import run as base_run
from .derive_labels_v3 import validate_control_source
from .label_control_binding import load_control_config


def run(config_path: Path, source_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 label-control result exists: {output}")
    root = Path(__file__).resolve().parents[2]
    control = load_control_config(root, config_path)
    source_rows = validate_control_source(root, config_path, source_run)
    base_config = root / str(control["base_config"]["path"])
    engine_dir = output / "engine"
    original_validator = base_run._validate_source

    def bound_validator(_root: Path, candidate_config: Path, candidate_source: Path):
        if sha256_file(candidate_config) != control["base_config"]["sha256"]:
            raise R14Error("R21 label-control run received a different base config")
        if candidate_source.resolve() != source_run.resolve():
            raise R14Error("R21 label-control run received a different source directory")
        return source_rows

    base_run._validate_source = bound_validator
    try:
        engine = base_run.run(base_config, source_run, engine_dir)
    finally:
        base_run._validate_source = original_validator

    result = {
        "format": "abi-r21-label-control-bakeoff-result/1",
        "verdict": engine["verdict"],
        "control_config_sha256": sha256_file(config_path),
        "base_config_sha256": control["base_config"]["sha256"],
        "control_source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "control_source_rows_sha256": sha256_file(source_run / "source_observations.jsonl"),
        "engine_result": {
            "path": "engine/result.json",
            "sha256": sha256_file(engine_dir / "result.json"),
            "evidence_sha256": engine["evidence_sha256"],
        },
        "response_regeneration_performed": False,
        "teacher_side_labeling": False,
        "autonomous_ontology_discovery": False,
        "scientific_claim": engine["claim"],
        "claim_ceiling": engine["claim_ceiling"],
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "result.json", result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.config, args.source_run, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
