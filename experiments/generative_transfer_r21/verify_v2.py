"""Strictly recompute a negative R21 result produced after label repair."""

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

from . import verify as base_verify
from .acquire_labels_v2 import validate_repaired_source
from .label_repair_binding import load_repair_config


def verify(config_path: Path, source_run: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    repair = load_repair_config(root, config_path)
    source_rows = validate_repaired_source(root, config_path, source_run)
    base_config = root / str(repair["base_config"]["path"])
    wrapper = json_object(result_dir / "result.json")
    engine_dir = result_dir / "engine"
    if (
        wrapper.get("format") != "abi-r21-label-repair-bakeoff-result/1"
        or wrapper.get("repair_config_sha256") != sha256_file(config_path)
        or wrapper.get("base_config_sha256") != repair["base_config"]["sha256"]
        or wrapper.get("repaired_source_receipt_sha256") != sha256_file(source_run / "receipt.json")
        or wrapper.get("repaired_source_rows_sha256")
        != sha256_file(source_run / "source_observations.jsonl")
        or wrapper.get("engine_result", {}).get("sha256") != sha256_file(engine_dir / "result.json")
        or wrapper.get("evidence_sha256") != evidence_hash(wrapper)
        or wrapper.get("response_regeneration_performed") is not False
        or wrapper.get("evaluation_access_during_label_repair") is not False
    ):
        raise R14Error("R21 repaired wrapper evidence failed")

    original_validator = base_verify._validate_source

    def bound_validator(_root: Path, candidate_config: Path, candidate_source: Path):
        if sha256_file(candidate_config) != repair["base_config"]["sha256"]:
            raise R14Error("R21 repaired verifier received a different base config")
        if candidate_source.resolve() != source_run.resolve():
            raise R14Error("R21 repaired verifier received a different source directory")
        return source_rows

    base_verify._validate_source = bound_validator
    try:
        engine_verification = base_verify.verify(base_config, source_run, engine_dir)
    finally:
        base_verify._validate_source = original_validator

    verification = {
        "format": "abi-r21-label-repair-strict-verification/1",
        "status": engine_verification["status"],
        "repair_config_sha256": sha256_file(config_path),
        "wrapper_result_sha256": sha256_file(result_dir / "result.json"),
        "engine_result_sha256": sha256_file(engine_dir / "result.json"),
        "engine_verification": engine_verification,
        "source_rows_revalidated": len(source_rows),
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = evidence_hash(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(args.config, args.source_run, args.result)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
