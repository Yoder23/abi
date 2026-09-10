"""Physically compile R19 on both disclosed development evidence sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.functional_realization_r18.run_public import (
    _compiler_records,
    _control,
    _independent_modal,
    _modal_realize,
    _rows,
)
from experiments.functional_realization_r18.verify_heldout_source import (
    verify_source as verify_r18_source,
)
from experiments.linguistic_realization_r17.verify_source import verify_source as verify_r17_source

from .isolation import R19IsolationError, run_wsl_isolated_extraction
from .package import load_package, realize


def _bundle(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    value = {"format": "abi-r19-anonymous-teacher-realizations/1", "records": records}
    value["evidence_sha256"] = evidence_hash(value)
    write_json_once(path, value)
    return value


def _compile_dataset(
    root: Path,
    source_rows: Path,
    destination: Path,
    shuffle_seed: int,
) -> dict[str, Any]:
    destination.mkdir(parents=True)
    rows = _rows(source_rows)
    records = _compiler_records(rows)
    random.Random(shuffle_seed).shuffle(records)
    bundle_path = destination / "source_bundle.json"
    _bundle(bundle_path, records)
    primary = run_wsl_isolated_extraction(root, bundle_path, destination / "extraction")
    control_bundle = destination / "mood_permutation_bundle.json"
    _bundle(control_bundle, _control(records))
    control = None
    control_rejection = None
    try:
        control = run_wsl_isolated_extraction(
            root, control_bundle, destination / "control_extraction"
        )
    except R19IsolationError as exc:
        if "no structure-compatible polarity contrast" not in str(exc):
            raise
        rejection_path = destination / "control_rejection.json"
        rejection = {
            "format": "abi-r19-control-rejection/1",
            "reason": "NO_STRUCTURE_COMPATIBLE_POLARITY_CONTRAST",
            "exception": str(exc),
            "source_bundle_sha256": sha256_file(control_bundle),
            "isolated_worker_sha256": sha256_file(
                root / "experiments/contrastive_realization_r19/isolated_worker.py"
            ),
            "packages_emitted": 0,
        }
        rejection["evidence_sha256"] = evidence_hash(rejection)
        write_json_once(rejection_path, rejection)
        control_rejection = {
            "status": "REJECTED_NO_STRUCTURE_COMPATIBLE_POLARITY_CONTRAST",
            "package_emitted": False,
            "runtime_behavior": "ABSTAIN",
            "evidence": {
                "path": rejection_path.name,
                "sha256": sha256_file(rejection_path),
            },
        }
    package_path = destination / "extraction" / primary["result"]["package"]["path"]
    package = load_package(package_path)
    control_path = None
    control_package = None
    if control is not None:
        control_path = destination / "control_extraction" / control["result"]["package"]["path"]
        control_package = load_package(control_path)
    modal = _independent_modal(records)
    evaluation = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        package_output = realize(package, row["signature"], row["slots"])
        control_output = realize(control_package, row["signature"], row["slots"])
        modal_output = _modal_realize(modal, row)
        evaluation.append(
            {
                "record_id": row["record_id"],
                "signature": row["signature"],
                "slots": row["slots"],
                "expected": row["expected"],
                "source_output": row["output"],
                "source_functional_exact": row["functional_exact"],
                "package_output": package_output,
                "package_functional_exact": package_output == row["expected"],
                "package_source_exact": package_output == row["output"],
                "modal_output": modal_output,
                "modal_functional_exact": modal_output == row["expected"],
                "control_output": control_output,
                "control_functional_exact": control_output == row["expected"],
                "removed_output": realize(None, row["signature"], row["slots"]),
            }
        )
    evaluation_path = destination / "evaluation.jsonl"
    write_jsonl_once(evaluation_path, evaluation)
    metrics = {
        "source_rows": len(rows),
        "extraction_rows": len(records),
        "evaluation_rows": len(evaluation),
        "source_functional_exact": sum(row["source_functional_exact"] for row in evaluation),
        "package_functional_exact": sum(row["package_functional_exact"] for row in evaluation),
        "package_source_exact": sum(row["package_source_exact"] for row in evaluation),
        "independent_modal_functional_exact": sum(
            row["modal_functional_exact"] for row in evaluation
        ),
        "source_exact_regressions": sum(
            row["source_functional_exact"] and not row["package_functional_exact"]
            for row in evaluation
        ),
        "control_functional_exact": sum(row["control_functional_exact"] for row in evaluation),
        "removed_abstain": sum(row["removed_output"] is None for row in evaluation),
    }
    exact = len(evaluation)
    passed = (
        len(rows) == 120
        and len(records) == 72
        and exact == 48
        and primary["result"]["templates_learned"] == 24
        and metrics["package_functional_exact"] == exact
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] >= metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == exact
        and metrics["control_functional_exact"] / exact <= 0.10
    )
    return {
        "verdict": "PASS" if passed else "FAIL",
        "metrics": metrics,
        "source_rows_sha256": sha256_file(source_rows),
        "source_bundle": {"path": str(bundle_path.name), "sha256": sha256_file(bundle_path)},
        "control_bundle": {
            "path": str(control_bundle.name),
            "sha256": sha256_file(control_bundle),
        },
        "evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path)},
        "package": {
            "path": str(package_path.relative_to(destination)),
            "bytes": package_path.stat().st_size,
            "sha256": sha256_file(package_path),
        },
        "control": control_rejection
        or {
            "status": "PACKAGE_EMITTED",
            "package": {
                "path": str(control_path.relative_to(destination)),
                "bytes": control_path.stat().st_size,
                "sha256": sha256_file(control_path),
            },
            "extraction_result_sha256": sha256_file(destination / "control_extraction/result.json"),
        },
        "extraction_result_sha256": sha256_file(destination / "extraction/result.json"),
    }


def run(
    r17_source: Path,
    r18_config: Path,
    r18_reveal: Path,
    r18_source: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R19 development output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    r17_strict = verify_r17_source(r17_source, "v2")
    if r17_strict != json_object(r17_source / "strict_source.json"):
        raise R14Error("R19 R17 source prerequisite changed")
    r18_strict = verify_r18_source(r18_config, r18_reveal, r18_source)
    if r18_strict != json_object(r18_source / "strict_verification.json"):
        raise R14Error("R19 R18 source prerequisite changed")
    output.mkdir(parents=True)
    datasets = {
        "r17_public_v2": _compile_dataset(
            root,
            r17_source / "source_observations.jsonl",
            output / "r17_public_v2",
            19_001,
        ),
        "r18_failed_hidden": _compile_dataset(
            root,
            r18_source / "source_observations.jsonl",
            output / "r18_failed_hidden",
            19_002,
        ),
    }
    passed = all(item["verdict"] == "PASS" for item in datasets.values())
    receipt = {
        "format": "abi-r19-disclosed-development-qualification/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": "DISCLOSED_DEVELOPMENT_CONTRASTIVE_REALIZATION_PREREQUISITE",
        "claim_ceiling": "NOT_HELD_OUT_OR_UNRESTRICTED_ENGLISH",
        "protocol_sha256": sha256_file(Path(__file__).with_name("PUBLIC_PROTOCOL_V4.md")),
        "datasets": datasets,
        "source_training_steps": 0,
        "host_training_steps": 0,
        "teacher_present_at_compilation": False,
        "teacher_present_at_package_execution": False,
        "physical_primary_extractions": 2,
        "physical_control_attempts": 2,
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r17-source", type=Path, required=True)
    parser.add_argument("--r18-config", type=Path, required=True)
    parser.add_argument("--r18-reveal", type=Path, required=True)
    parser.add_argument("--r18-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.r17_source,
                args.r18_config,
                args.r18_reveal,
                args.r18_source,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
