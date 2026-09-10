"""Fail-closed strict verification of R19 hidden replication."""

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

from .heldout_protocol import load_bound_inputs
from .verify_development import _verified, _verify_dataset
from .verify_heldout_source import verify_source


def verify(
    config_path: Path,
    reveal_path: Path,
    source_run: Path,
    run_dir: Path,
    *,
    _preverified_source: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    load_bound_inputs(root, config_path, reveal_path)
    if not _preverified_source:
        strict_source = verify_source(config_path, reveal_path, source_run)
        if strict_source != json_object(source_run / "strict_verification.json"):
            raise R14Error("R19 hidden source verification changed")
    receipt = _verified(run_dir / "receipt.json", "hidden receipt")
    if (
        receipt.get("format") != "abi-r19-heldout-contrastive-realization/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("reveal_sha256") != sha256_file(reveal_path)
        or receipt.get("protocol_sha256")
        != sha256_file(Path(__file__).with_name("HOLDOUT_PROTOCOL.md"))
        or receipt.get("source_receipt_sha256") != sha256_file(source_run / "receipt.json")
        or receipt.get("source_strict_sha256")
        != sha256_file(source_run / "strict_verification.json")
        or receipt.get("source_training_steps") != 0
        or receipt.get("host_training_steps") != 0
        or receipt.get("teacher_present_at_compilation") is not False
        or receipt.get("teacher_present_at_package_execution") is not False
        or receipt.get("physical_primary_extractions") != 1
        or receipt.get("physical_control_attempts") != 1
        or receipt.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R19 hidden receipt changed")
    metrics = _verify_dataset(
        root,
        run_dir / "hidden",
        source_run / "source_observations.jsonl",
        19_003,
        receipt["dataset"],
        require_pass=False,
    )
    gates = json_object(config_path)["gates"]
    passed = (
        metrics["package_functional_exact"] == gates["package_functional_exact"]
        and metrics["source_exact_regressions"] == gates["max_source_exact_regressions"]
        and (
            not gates["require_package_at_least_modal"]
            or metrics["package_functional_exact"] >= metrics["independent_modal_functional_exact"]
        )
        and metrics["control_functional_exact"] / metrics["evaluation_rows"]
        <= gates["max_control_accuracy"]
        and metrics["removed_abstain"] == gates["removed_abstain"]
    )
    if receipt.get("verdict") != ("PASS" if passed else "FAIL"):
        raise R14Error("R19 hidden verdict changed")
    expected_claim = (
        "BOUNDED_REPLICATED_CONTRASTIVE_ENGLISH_REALIZATION"
        if passed
        else "R19_HELDOUT_CONTRASTIVE_REALIZATION_FAILED"
    )
    if receipt.get("claim") != expected_claim:
        raise R14Error("R19 hidden claim changed")
    source_receipt = json_object(source_run / "receipt.json")
    accounting = receipt.get("information_accounting", {})
    inherited = source_receipt.get("information_accounting", {})
    package_path = run_dir / "hidden" / receipt["dataset"]["package"]["path"]
    bundle_path = run_dir / "hidden" / receipt["dataset"]["source_bundle"]["path"]
    compiled_evidence_bytes = sum(
        path.stat().st_size for path in (run_dir / "hidden").rglob("*") if path.is_file()
    )
    if (
        any(accounting.get(key) != value for key, value in inherited.items())
        or accounting.get("source_acquisition_receipt_bytes")
        != (source_run / "receipt.json").stat().st_size
        or accounting.get("source_strict_evidence_bytes")
        != (source_run / "strict_verification.json").stat().st_size
        or accounting.get("compiler_source_records") != 72
        or accounting.get("compiler_source_bundle_bytes") != bundle_path.stat().st_size
        or accounting.get("compiler_evidence_bytes_excluding_receipt") != compiled_evidence_bytes
        or accounting.get("compiler_elapsed_seconds", 0) <= 0
        or accounting.get("compiler_cpu_process_seconds", 0) <= 0
        or accounting.get("compiler_cpu_process_hours")
        != accounting.get("compiler_cpu_process_seconds") / 3600
        or accounting.get("final_package_bytes") != package_path.stat().st_size
        or accounting.get("final_templates") != 24
        or any(
            accounting.get(key) != 0
            for key in (
                "frozen_source_parameters_in_package",
                "host_parameters_in_package",
                "source_training_steps",
                "host_training_steps",
                "bridge_parameters_trained",
            )
        )
    ):
        raise R14Error("R19 hidden information accounting changed")
    result = {
        "format": "abi-r19-heldout-strict-verification/1",
        "verdict": receipt["verdict"],
        "claim": receipt["claim"],
        "claim_ceiling": receipt["claim_ceiling"],
        "source_rows_verified": metrics["source_rows"],
        "extraction_rows_verified": metrics["extraction_rows"],
        "evaluation_rows_verified": metrics["evaluation_rows"],
        "package_functional_exact": metrics["package_functional_exact"],
        "source_functional_exact": metrics["source_functional_exact"],
        "modal_functional_exact": metrics["independent_modal_functional_exact"],
        "source_exact_regressions": metrics["source_exact_regressions"],
        "control_functional_exact": metrics["control_functional_exact"],
        "teacher_present_at_package_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reveal", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.reveal, args.source_run, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
