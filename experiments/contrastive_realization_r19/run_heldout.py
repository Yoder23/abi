"""Compile and evaluate the fresh preregistered R19 hidden replication."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
)

from .heldout_protocol import load_bound_inputs
from .run_development import _compile_dataset
from .verify_heldout_source import verify_source


def run(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    output: Path,
) -> dict[str, object]:
    if output.exists():
        raise R14Error(f"immutable R19 held-out output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    load_bound_inputs(root, config_path, reveal_path)
    strict_source = verify_source(config_path, reveal_path, source_run)
    if (
        strict_source != json_object(source_run / "strict_verification.json")
        or not strict_source["compiler_authorized"]
    ):
        raise R14Error("R19 hidden source did not authorize compilation")
    output.mkdir(parents=True)
    compilation_started = time.perf_counter()
    compilation_cpu_started = time.process_time()
    dataset = _compile_dataset(
        root,
        source_run / "source_observations.jsonl",
        output / "hidden",
        19_003,
    )
    compilation_seconds = time.perf_counter() - compilation_started
    compilation_cpu_seconds = time.process_time() - compilation_cpu_started
    source_receipt = json_object(source_run / "receipt.json")
    compiled_evidence_bytes = sum(
        path.stat().st_size for path in (output / "hidden").rglob("*") if path.is_file()
    )
    accounting = {
        **source_receipt["information_accounting"],
        "source_acquisition_receipt_bytes": (source_run / "receipt.json").stat().st_size,
        "source_strict_evidence_bytes": (source_run / "strict_verification.json").stat().st_size,
        "compiler_source_records": dataset["metrics"]["extraction_rows"],
        "compiler_source_bundle_bytes": (output / "hidden" / dataset["source_bundle"]["path"])
        .stat()
        .st_size,
        "compiler_evidence_bytes_excluding_receipt": compiled_evidence_bytes,
        "compiler_elapsed_seconds": compilation_seconds,
        "compiler_cpu_process_seconds": compilation_cpu_seconds,
        "compiler_cpu_process_hours": compilation_cpu_seconds / 3600,
        "final_package_bytes": dataset["package"]["bytes"],
        "final_templates": 24,
        "frozen_source_parameters_in_package": 0,
        "host_parameters_in_package": 0,
        "source_training_steps": 0,
        "host_training_steps": 0,
        "bridge_parameters_trained": 0,
    }
    passed = dataset["verdict"] == "PASS"
    receipt: dict[str, object] = {
        "format": "abi-r19-heldout-contrastive-realization/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": (
            "BOUNDED_REPLICATED_CONTRASTIVE_ENGLISH_REALIZATION"
            if passed
            else "R19_HELDOUT_CONTRASTIVE_REALIZATION_FAILED"
        ),
        "claim_ceiling": "NOT_UNRESTRICTED_ENGLISH_OR_LAYERCAKE_ACCEPTANCE",
        "config_sha256": sha256_file(config_path),
        "reveal_sha256": sha256_file(reveal_path),
        "protocol_sha256": sha256_file(Path(__file__).with_name("HOLDOUT_PROTOCOL.md")),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_strict_sha256": sha256_file(source_run / "strict_verification.json"),
        "dataset": dataset,
        "source_training_steps": 0,
        "host_training_steps": 0,
        "teacher_present_at_compilation": False,
        "teacher_present_at_package_execution": False,
        "physical_primary_extractions": 1,
        "physical_control_attempts": 1,
        "information_accounting": accounting,
        "full_abi_moonshot": "OPEN",
        "unproven": [
            "unrestricted English fluency",
            "prompt understanding or conversation",
            "autonomous capability discovery or labeling",
            "arbitrary-domain extraction",
            "production LayerCake ingestion",
            "global minimality",
            "superiority to LoRA or distillation",
        ],
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.reveal, args.source_run, args.output), indent=2))


if __name__ == "__main__":
    main()
