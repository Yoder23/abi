"""Fail-closed recomputation of the R20 public experiment."""

from __future__ import annotations

import argparse
import hashlib
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
)
from experiments.preexisting_representation_r15b.public_qualification import canonical_json_bytes

from .binding import load_config
from .compiler import infer_package
from .package import execute, load_package
from .protocol import TASKS
from .run_public import _control_records, _paired_bootstrap, _rows
from .source_acquisition import _compiler_records
from .verify_source import verify as verify_source


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    if value.get("evidence_sha256") != evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    ):
        raise R14Error(f"R20 {name} evidence hash changed")
    return value


def _verify_bundle(path: Path, records: list[dict[str, str]], expected_sha: str) -> dict[str, Any]:
    value = _verified(path, "bundle")
    if (
        sha256_file(path) != expected_sha
        or value.get("format") != "abi-r20-anonymous-instruction-realizations/1"
        or value.get("records") != records
        or any(
            set(record) != {"record_id", "prompt", "output"}
            or {
                "task",
                "expected",
                "oracle",
                "evaluation",
                "secret",
                "success_id",
            }.intersection(record)
            for record in value.get("records", [])
        )
    ):
        raise R14Error("R20 source bundle changed")
    return value


def _spec_sha256() -> str:
    value = {
        "format": "abi-r20-instructional-compiler/1",
        "expected_classes": 6,
        "records": 72,
        "task_labels": 0,
        "expected_outputs": 0,
        "evaluation_rows": 0,
        "success_ids": 0,
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return hashlib.sha256(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n").hexdigest()


def _verify_extraction(
    root: Path,
    run_dir: Path,
    name: str,
    bundle_path: Path,
    bundle: dict[str, Any],
    expected_package: dict[str, Any],
    expected_diagnostics: dict[str, Any],
) -> Path:
    directory = run_dir / name
    result = _verified(directory / "result.json", "extraction result")
    launcher = _verified(directory / "launcher.json", "extraction launcher")
    manifest = _verified(directory / "manifest.json", "extraction manifest")
    mountinfo = directory / "mountinfo.txt"
    if (
        result.get("format") != "abi-r20-isolated-instructional-extraction/1"
        or result.get("records_consumed") != 72
        or result.get("programs_selected") != 6
        or result.get("task_labels_consumed") != 0
        or result.get("expected_outputs_consumed") != 0
        or result.get("evaluation_rows_consumed") != 0
        or result.get("training_steps") != 0
        or any(result.get(key) != value for key, value in expected_diagnostics.items())
        or result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("network_namespace_isolated") is not True
        or result.get("bundle_evidence_sha256") != bundle["evidence_sha256"]
    ):
        raise R14Error("R20 physical extraction claims changed")
    if (
        launcher.get("format") != "abi-r20-isolated-extraction-launcher/1"
        or launcher.get("worker_exit_code") != 0
        or launcher.get("sandbox_policy") != "linux-pivot-root-no-network/1"
        or launcher.get("result_sha256") != sha256_file(directory / "result.json")
        or launcher.get("mountinfo_sha256") != sha256_file(mountinfo)
        or result.get("mountinfo_sha256") != sha256_file(mountinfo)
    ):
        raise R14Error("R20 extraction launcher changed")
    mount_text = mountinfo.read_text(encoding="utf-8")
    if (
        " /capsule " not in mount_text
        or " /proc " not in mount_text
        or " /mnt/c " in mount_text.casefold()
        or " /oldroot " in mount_text.casefold()
    ):
        raise R14Error("R20 physical mount evidence changed")
    files = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    if (
        manifest.get("format") != "abi-r20-isolated-instructional-capsule/1"
        or files
        != {
            "isolated_worker.py": sha256_file(
                root / "experiments/instructional_realization_r20/isolated_worker.py"
            ),
            "source_bundle.json": sha256_file(bundle_path),
            "spec.json": _spec_sha256(),
        }
        or manifest.get("prompt_rows_included") != 72
        or any(
            manifest.get(key) != 0
            for key in (
                "task_labels_included",
                "expected_outputs_included",
                "evaluation_rows_included",
                "success_ids_included",
            )
        )
        or manifest.get("network") is not False
        or result.get("capsule_manifest_evidence_sha256") != manifest["evidence_sha256"]
        or launcher.get("manifest_evidence_sha256") != manifest["evidence_sha256"]
    ):
        raise R14Error("R20 extraction manifest changed")
    package_ref = result.get("package", {})
    package_path = directory / str(package_ref.get("path"))
    package = load_package(package_path)
    expected_bytes = (
        json.dumps(expected_package, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    )
    expected_sha = hashlib.sha256(expected_bytes).hexdigest()
    if (
        package != expected_package
        or package_path.read_bytes() != expected_bytes
        or package_ref.get("bytes") != len(expected_bytes)
        or package_ref.get("sha256") != expected_sha
        or package_path.name != f"sha256-{expected_sha}.abipkg"
    ):
        raise R14Error("R20 package changed")
    return package_path


def verify(
    config_path: Path,
    source_run: Path,
    run_dir: Path,
    *,
    _preverified_source: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    load_config(root, config_path)
    if not _preverified_source:
        source_strict = verify_source(config_path, source_run)
        if source_strict != json_object(source_run / "strict_verification.json"):
            raise R14Error("R20 source verification changed")
    receipt = _verified(run_dir / "receipt.json", "receipt")
    if (
        receipt.get("format") != "abi-r20-public-instructional-realization/1"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("protocol_sha256")
        != sha256_file(root / "experiments/instructional_realization_r20/PUBLIC_PROTOCOL.md")
        or receipt.get("source_receipt_sha256") != sha256_file(source_run / "receipt.json")
        or receipt.get("source_strict_sha256")
        != sha256_file(source_run / "strict_verification.json")
        or receipt.get("source_training_steps") != 0
        or receipt.get("host_training_steps") != 0
        or receipt.get("teacher_present_at_compilation") is not False
        or receipt.get("teacher_present_at_package_execution") is not False
        or receipt.get("physical_primary_extractions") != 1
        or receipt.get("physical_control_extractions") != 1
        or receipt.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R20 receipt changed")
    rows_path = source_run / "source_observations.jsonl"
    rows = _rows(rows_path)
    primary_records = _compiler_records(rows)
    random.Random(20_001).shuffle(primary_records)
    control_records = _control_records(rows)
    random.Random(20_002).shuffle(control_records)
    artifacts = receipt["artifacts"]
    source_bundle_path = run_dir / artifacts["source_bundle"]["path"]
    control_bundle_path = run_dir / artifacts["control_bundle"]["path"]
    source_bundle = _verify_bundle(
        source_bundle_path, primary_records, artifacts["source_bundle"]["sha256"]
    )
    control_bundle = _verify_bundle(
        control_bundle_path, control_records, artifacts["control_bundle"]["sha256"]
    )
    primary_expected, primary_diagnostics = infer_package(primary_records)
    control_expected, control_diagnostics = infer_package(control_records)
    package_path = _verify_extraction(
        root,
        run_dir,
        "extraction",
        source_bundle_path,
        source_bundle,
        primary_expected,
        primary_diagnostics,
    )
    control_path = _verify_extraction(
        root,
        run_dir,
        "control_extraction",
        control_bundle_path,
        control_bundle,
        control_expected,
        control_diagnostics,
    )
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
                "removed_output": None,
            }
        )
    evaluation_path = run_dir / artifacts["evaluation"]["path"]
    try:
        stored_rows = [
            json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R14Error("R20 evaluation evidence unavailable") from exc
    if (
        sha256_file(evaluation_path) != artifacts["evaluation"]["sha256"]
        or stored_rows != evaluation
    ):
        raise R14Error("R20 evaluation evidence changed")
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
        "removed_abstain": len(evaluation),
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
    if (
        receipt.get("metrics") != metrics
        or receipt.get("verdict") != ("PASS" if passed else "FAIL")
        or receipt.get("claim")
        != (
            "PUBLIC_INSTRUCTIONAL_REALIZATION_PREREQUISITE_PASSED"
            if passed
            else "PUBLIC_INSTRUCTIONAL_REALIZATION_PREREQUISITE_FAILED"
        )
        or artifacts.get("source_rows_sha256") != sha256_file(rows_path)
        or artifacts.get("primary_result_sha256") != sha256_file(run_dir / "extraction/result.json")
        or artifacts.get("control_result_sha256")
        != sha256_file(run_dir / "control_extraction/result.json")
        or receipt["package"]
        != {
            "path": str(package_path.relative_to(run_dir)),
            "bytes": package_path.stat().st_size,
            "sha256": sha256_file(package_path),
        }
        or receipt["control_package"]
        != {
            "path": str(control_path.relative_to(run_dir)),
            "bytes": control_path.stat().st_size,
            "sha256": sha256_file(control_path),
        }
    ):
        raise R14Error("R20 scientific evidence changed")
    accounting = receipt.get("information_accounting", {})
    inherited = json_object(source_run / "receipt.json")["information_accounting"]
    evidence_bytes = sum(
        path.stat().st_size
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.relative_to(run_dir).as_posix()
        not in {"receipt.json", "strict_verification.json", "hostile_audit.json"}
    )
    if (
        any(accounting.get(key) != value for key, value in inherited.items())
        or accounting.get("compiler_source_records") != 72
        or accounting.get("compiler_source_bundle_bytes") != source_bundle_path.stat().st_size
        or accounting.get("compiler_evidence_bytes_excluding_receipt") != evidence_bytes
        or accounting.get("compiler_elapsed_seconds", 0) <= 0
        or accounting.get("compiler_cpu_process_seconds", 0) <= 0
        or accounting.get("compiler_cpu_process_hours")
        != accounting.get("compiler_cpu_process_seconds") / 3600
        or accounting.get("final_package_bytes") != package_path.stat().st_size
        or accounting.get("final_programs") != 6
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
        raise R14Error("R20 information accounting changed")
    result = {
        "format": "abi-r20-public-strict-verification/1",
        "verdict": receipt["verdict"],
        "claim": receipt["claim"],
        "source_rows_verified": len(rows),
        "evaluation_rows_verified": len(evaluation),
        "package_functional_exact": metrics["package_functional_exact"],
        "source_functional_exact": metrics["source_functional_exact"],
        "control_functional_exact": metrics["control_functional_exact"],
        "source_exact_regressions": metrics["source_exact_regressions"],
        "paired_bootstrap_95ci": metrics["paired_bootstrap_95ci"],
        "teacher_present_at_package_execution": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config, args.source_run, args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
