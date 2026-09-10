"""Strict R21 verification with the additive self-hash assurance repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
)

from . import derive_labels_v3, verify_v3
from . import verify as base_verify
from .hash_assurance_binding import load_assurance_config, selfless_evidence_hash


def verify(config_path: Path, source_run: Path, result_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    config = load_assurance_config(root, config_path)
    control_config = root / str(config["label_control_config"]["path"])
    assurance = json_object(result_dir / "assurance_binding.json")
    if (
        assurance.get("format") != "abi-r21-hash-assurance-run/1"
        or assurance.get("assurance_config_sha256") != sha256_file(config_path)
        or assurance.get("result_sha256") != sha256_file(result_dir / "result.json")
        or assurance.get("evidence_sha256") != selfless_evidence_hash(assurance)
        or assurance.get("data_changes_performed") is not False
        or assurance.get("gate_changes_performed") is not False
    ):
        raise R14Error("R21 assurance-run binding failed")

    original_derive_hash = derive_labels_v3.evidence_hash
    original_wrapper_hash = verify_v3.evidence_hash
    original_engine_hash = base_verify.evidence_hash
    derive_labels_v3.evidence_hash = selfless_evidence_hash
    verify_v3.evidence_hash = selfless_evidence_hash
    base_verify.evidence_hash = selfless_evidence_hash
    try:
        underlying = verify_v3.verify(control_config, source_run, result_dir)
    finally:
        derive_labels_v3.evidence_hash = original_derive_hash
        verify_v3.evidence_hash = original_wrapper_hash
        base_verify.evidence_hash = original_engine_hash

    verification = {
        "format": "abi-r21-hash-assurance-strict-verification/1",
        "status": underlying["status"],
        "assurance_config_sha256": sha256_file(config_path),
        "assurance_binding_sha256": sha256_file(result_dir / "assurance_binding.json"),
        "underlying_verification": underlying,
        "removed_self_hash_fields": ["evidence_sha256"],
        "data_changes": 0,
        "gate_changes": 0,
        "full_abi_moonshot": "OPEN",
    }
    verification["evidence_sha256"] = selfless_evidence_hash(verification)
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
