"""Compile and evaluate the R20 public instruction-conditioned package."""

from __future__ import annotations

import argparse
import json
import random
import time
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

from .binding import load_config
from .isolation import run_wsl_isolated_extraction
from .package import execute, load_package
from .protocol import TASKS, replace_instruction
from .source_acquisition import _compiler_records
from .verify_source import verify as verify_source


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error("R20 source rows unavailable") from exc


def _bundle(path: Path, records: list[dict[str, str]]) -> dict[str, Any]:
    value = {"format": "abi-r20-anonymous-instruction-realizations/1", "records": records}
    value["evidence_sha256"] = evidence_hash(value)
    write_json_once(path, value)
    return value


def _control_records(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    extraction = [row for row in rows if row["split"] == "extraction"]
    instructions = {
        task: [row["instruction"] for row in extraction if row["task"] == task] for task in TASKS
    }
    counters = {task: 0 for task in TASKS}
    records = []
    for row in extraction:
        task = str(row["task"])
        replacement_task = TASKS[(TASKS.index(task) + 1) % len(TASKS)]
        index = counters[task]
        counters[task] += 1
        records.append(
            {
                "record_id": str(row["record_id"]),
                "prompt": replace_instruction(
                    str(row["prompt"]), instructions[replacement_task][index]
                ),
                "output": str(row["output"]),
            }
        )
    return records


def _paired_bootstrap(values: list[int], seed: int = 20_001) -> list[float]:
    generator = random.Random(seed)
    estimates = []
    for _ in range(10_000):
        estimates.append(
            sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        )
    estimates.sort()
    return [estimates[249], estimates[9749]]


def run(config_path: Path, source_run: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R20 public output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    load_config(root, config_path)
    source_strict = verify_source(config_path, source_run)
    if (
        source_strict != json_object(source_run / "strict_verification.json")
        or not source_strict["compiler_authorized"]
    ):
        raise R14Error("R20 source did not authorize compilation")
    output.mkdir(parents=True)
    rows_path = source_run / "source_observations.jsonl"
    rows = _rows(rows_path)
    primary_records = _compiler_records(rows)
    random.Random(20_001).shuffle(primary_records)
    control_records = _control_records(rows)
    random.Random(20_002).shuffle(control_records)
    source_bundle = output / "source_bundle.json"
    control_bundle = output / "rotated_instruction_bundle.json"
    _bundle(source_bundle, primary_records)
    _bundle(control_bundle, control_records)
    compilation_started = time.perf_counter()
    compilation_cpu_started = time.process_time()
    primary = run_wsl_isolated_extraction(root, source_bundle, output / "extraction")
    control = run_wsl_isolated_extraction(root, control_bundle, output / "control_extraction")
    compilation_seconds = time.perf_counter() - compilation_started
    compilation_cpu_seconds = time.process_time() - compilation_cpu_started
    package_path = output / "extraction" / primary["result"]["package"]["path"]
    control_path = output / "control_extraction" / control["result"]["package"]["path"]
    package = load_package(package_path)
    control_package = load_package(control_path)
    evaluation = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        package_output = execute(package, str(row["prompt"]))
        control_output = execute(control_package, str(row["prompt"]))
        evaluation.append(
            {
                "record_id": row["record_id"],
                "task": row["task"],
                "prompt": row["prompt"],
                "expected": row["expected"],
                "source_output": row["output"],
                "source_functional_exact": row["functional_exact"],
                "package_output": package_output,
                "package_functional_exact": package_output == row["expected"],
                "package_source_exact": package_output == row["output"],
                "control_output": control_output,
                "control_functional_exact": control_output == row["expected"],
                "removed_output": execute(None, str(row["prompt"])),
            }
        )
    evaluation_path = output / "evaluation.jsonl"
    write_jsonl_once(evaluation_path, evaluation)
    package_by_task = {
        task: sum(row["package_functional_exact"] for row in evaluation if row["task"] == task)
        for task in TASKS
    }
    metrics = {
        "source_rows": len(rows),
        "extraction_rows": len(primary_records),
        "evaluation_rows": len(evaluation),
        "source_functional_exact": sum(row["source_functional_exact"] for row in evaluation),
        "package_functional_exact": sum(row["package_functional_exact"] for row in evaluation),
        "package_source_exact": sum(row["package_source_exact"] for row in evaluation),
        "package_exact_by_task": package_by_task,
        "source_exact_regressions": sum(
            row["source_functional_exact"] and not row["package_functional_exact"]
            for row in evaluation
        ),
        "control_functional_exact": sum(row["control_functional_exact"] for row in evaluation),
        "removed_abstain": sum(row["removed_output"] is None for row in evaluation),
        "paired_package_minus_source": (
            sum(row["package_functional_exact"] for row in evaluation)
            - sum(row["source_functional_exact"] for row in evaluation)
        )
        / len(evaluation),
        "paired_bootstrap_95ci": _paired_bootstrap(
            [
                int(row["package_functional_exact"]) - int(row["source_functional_exact"])
                for row in evaluation
            ]
        ),
    }
    passed = (
        metrics["package_functional_exact"] == 120
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] >= metrics["source_functional_exact"]
        and min(package_by_task.values()) >= 19
        and metrics["control_functional_exact"] <= 24
        and metrics["removed_abstain"] == 120
    )
    source_receipt = json_object(source_run / "receipt.json")
    evidence_bytes = sum(
        path.stat().st_size
        for path in output.rglob("*")
        if path.is_file()
        and path.relative_to(output).as_posix()
        not in {"receipt.json", "strict_verification.json", "hostile_audit.json"}
    )
    receipt = {
        "format": "abi-r20-public-instructional-realization/1",
        "verdict": "PASS" if passed else "FAIL",
        "claim": (
            "PUBLIC_INSTRUCTIONAL_REALIZATION_PREREQUISITE_PASSED"
            if passed
            else "PUBLIC_INSTRUCTIONAL_REALIZATION_PREREQUISITE_FAILED"
        ),
        "claim_ceiling": "NOT_HELD_OUT_OR_UNRESTRICTED_ENGLISH",
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(Path(__file__).with_name("PUBLIC_PROTOCOL.md")),
        "source_receipt_sha256": sha256_file(source_run / "receipt.json"),
        "source_strict_sha256": sha256_file(source_run / "strict_verification.json"),
        "metrics": metrics,
        "artifacts": {
            "source_rows_sha256": sha256_file(rows_path),
            "source_bundle": {"path": source_bundle.name, "sha256": sha256_file(source_bundle)},
            "control_bundle": {"path": control_bundle.name, "sha256": sha256_file(control_bundle)},
            "evaluation": {"path": evaluation_path.name, "sha256": sha256_file(evaluation_path)},
            "primary_result_sha256": sha256_file(output / "extraction/result.json"),
            "control_result_sha256": sha256_file(output / "control_extraction/result.json"),
        },
        "package": {
            "path": str(package_path.relative_to(output)),
            "bytes": package_path.stat().st_size,
            "sha256": sha256_file(package_path),
        },
        "control_package": {
            "path": str(control_path.relative_to(output)),
            "bytes": control_path.stat().st_size,
            "sha256": sha256_file(control_path),
        },
        "information_accounting": {
            **source_receipt["information_accounting"],
            "compiler_source_records": len(primary_records),
            "compiler_source_bundle_bytes": source_bundle.stat().st_size,
            "compiler_evidence_bytes_excluding_receipt": evidence_bytes,
            "compiler_elapsed_seconds": compilation_seconds,
            "compiler_cpu_process_seconds": compilation_cpu_seconds,
            "compiler_cpu_process_hours": compilation_cpu_seconds / 3600,
            "final_package_bytes": package_path.stat().st_size,
            "final_programs": 6,
            "frozen_source_parameters_in_package": 0,
            "host_parameters_in_package": 0,
            "source_training_steps": 0,
            "host_training_steps": 0,
            "bridge_parameters_trained": 0,
        },
        "source_training_steps": 0,
        "host_training_steps": 0,
        "teacher_present_at_compilation": False,
        "teacher_present_at_package_execution": False,
        "physical_primary_extractions": 1,
        "physical_control_extractions": 1,
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.source_run, args.output), indent=2))


if __name__ == "__main__":
    main()
