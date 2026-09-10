"""Fail-closed strict verification for R19 disclosed development evidence."""

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
from experiments.linguistic_realization_r17.frames import SIGNATURES
from experiments.linguistic_realization_r17.verify_source import verify_source as verify_r17_source

from .compiler import infer_templates
from .package import PACKAGE_FORMAT, load_package, realize


def _verified(path: Path, name: str) -> dict[str, Any]:
    value = json_object(path)
    stored = value.get("evidence_sha256")
    if stored != evidence_hash(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    ):
        raise R14Error(f"R19 {name} evidence hash changed")
    return value


def _bundle(path: Path, expected: list[dict[str, Any]], expected_sha: str) -> dict[str, Any]:
    value = _verified(path, "bundle")
    forbidden = {
        "prompt",
        "expected",
        "oracle",
        "evaluator",
        "evaluation",
        "secret",
        "reveal",
        "success_id",
    }
    if (
        sha256_file(path) != expected_sha
        or value.get("format") != "abi-r19-anonymous-teacher-realizations/1"
        or value.get("records") != expected
        or any(forbidden.intersection(record) for record in value.get("records", []))
    ):
        raise R14Error("R19 bundle changed")
    return value


def _package(records: list[dict[str, Any]]) -> dict[str, Any]:
    templates, diagnostics = infer_templates(records)
    return {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "factorization": "same-number-polarity-contrast",
        "templates": templates,
        "support": diagnostics["support"],
    }


def _spec_sha256() -> str:
    spec = {
        "format": "abi-r19-contrastive-template-compiler/1",
        "signatures": list(SIGNATURES),
        "factorization": "same-number-polarity-contrast",
        "oracle_fields": 0,
        "evaluation_rows": 0,
        "success_ids": 0,
    }
    spec["evidence_sha256"] = evidence_hash(spec)
    return hashlib.sha256(json.dumps(spec, indent=2, sort_keys=True).encode() + b"\n").hexdigest()


def _verify_primary(
    root: Path,
    directory: Path,
    source_bundle: Path,
    bundle: dict[str, Any],
    expected_package: dict[str, Any],
) -> Path:
    extraction = directory / "extraction"
    result = _verified(extraction / "result.json", "extraction result")
    launcher = _verified(extraction / "launcher.json", "extraction launcher")
    manifest = _verified(extraction / "manifest.json", "extraction manifest")
    mountinfo = extraction / "mountinfo.txt"
    if (
        result.get("format") != "abi-r19-isolated-contrastive-extraction/1"
        or result.get("records_consumed") != 72
        or result.get("templates_learned") != 24
        or result.get("oracle_fields_consumed") != 0
        or result.get("evaluation_rows_consumed") != 0
        or result.get("old_root_present") is not False
        or result.get("windows_mount_present") is not False
        or result.get("network_namespace_isolated") is not True
        or result.get("bundle_evidence_sha256") != bundle["evidence_sha256"]
    ):
        raise R14Error("R19 physical extraction claims changed")
    if (
        launcher.get("format") != "abi-r19-isolated-extraction-launcher/1"
        or launcher.get("worker_exit_code") != 0
        or launcher.get("sandbox_policy") != "linux-pivot-root-no-network/1"
        or launcher.get("result_sha256") != sha256_file(extraction / "result.json")
        or launcher.get("mountinfo_sha256") != sha256_file(mountinfo)
        or result.get("mountinfo_sha256") != sha256_file(mountinfo)
    ):
        raise R14Error("R19 extraction launcher changed")
    mount_text = mountinfo.read_text(encoding="utf-8")
    if (
        " /capsule " not in mount_text
        or " /proc " not in mount_text
        or " /mnt/c " in mount_text.casefold()
        or " /oldroot " in mount_text.casefold()
    ):
        raise R14Error("R19 physical mount evidence changed")
    files = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    if (
        manifest.get("format") != "abi-r19-isolated-contrastive-capsule/1"
        or files
        != {
            "isolated_worker.py": sha256_file(
                root / "experiments/contrastive_realization_r19/isolated_worker.py"
            ),
            "source_bundle.json": sha256_file(source_bundle),
            "spec.json": _spec_sha256(),
        }
        or any(
            manifest.get(key) != 0
            for key in (
                "prompts_included",
                "oracle_fields_included",
                "evaluation_rows_included",
                "success_ids_included",
            )
        )
        or manifest.get("network") is not False
        or result.get("capsule_manifest_evidence_sha256") != manifest["evidence_sha256"]
        or launcher.get("manifest_evidence_sha256") != manifest["evidence_sha256"]
    ):
        raise R14Error("R19 extraction manifest changed")
    package_ref = result.get("package", {})
    package_path = extraction / str(package_ref.get("path"))
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
        raise R14Error("R19 package changed")
    return package_path


def _verify_control(
    root: Path,
    directory: Path,
    control_bundle: Path,
    receipt_control: dict[str, Any],
) -> None:
    rejection_path = directory / str(receipt_control["evidence"]["path"])
    rejection = _verified(rejection_path, "control rejection")
    if (
        receipt_control
        != {
            "status": "REJECTED_NO_STRUCTURE_COMPATIBLE_POLARITY_CONTRAST",
            "package_emitted": False,
            "runtime_behavior": "ABSTAIN",
            "evidence": {
                "path": rejection_path.name,
                "sha256": sha256_file(rejection_path),
            },
        }
        or rejection.get("format") != "abi-r19-control-rejection/1"
        or rejection.get("reason") != "NO_STRUCTURE_COMPATIBLE_POLARITY_CONTRAST"
        or "no structure-compatible polarity contrast" not in rejection.get("exception", "")
        or rejection.get("source_bundle_sha256") != sha256_file(control_bundle)
        or rejection.get("isolated_worker_sha256")
        != sha256_file(root / "experiments/contrastive_realization_r19/isolated_worker.py")
        or rejection.get("packages_emitted") != 0
        or (directory / "control_extraction").exists()
    ):
        raise R14Error("R19 control rejection evidence changed")


def _verify_dataset(
    root: Path,
    directory: Path,
    source_rows: Path,
    seed: int,
    stored: dict[str, Any],
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    rows = _rows(source_rows)
    records = _compiler_records(rows)
    random.Random(seed).shuffle(records)
    control_records = _control(records)
    source_bundle_path = directory / stored["source_bundle"]["path"]
    control_bundle_path = directory / stored["control_bundle"]["path"]
    source_bundle = _bundle(source_bundle_path, records, stored["source_bundle"]["sha256"])
    _bundle(
        control_bundle_path,
        control_records,
        stored["control_bundle"]["sha256"],
    )
    package_path = _verify_primary(
        root, directory, source_bundle_path, source_bundle, _package(records)
    )
    _verify_control(root, directory, control_bundle_path, stored["control"])
    package = load_package(package_path)
    modal = _independent_modal(records)
    evaluation = []
    for row in rows:
        if row["split"] != "evaluation":
            continue
        package_output = realize(package, row["signature"], row["slots"])
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
                "control_output": None,
                "control_functional_exact": False,
                "removed_output": None,
            }
        )
    evaluation_path = directory / stored["evaluation"]["path"]
    actual_rows = [
        json.loads(line) for line in evaluation_path.read_text(encoding="utf-8").splitlines()
    ]
    if sha256_file(evaluation_path) != stored["evaluation"]["sha256"] or actual_rows != evaluation:
        raise R14Error("R19 evaluation evidence changed")
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
        "control_functional_exact": 0,
        "removed_abstain": len(evaluation),
    }
    passed = (
        metrics["package_functional_exact"] == 48
        and metrics["source_exact_regressions"] == 0
        and metrics["package_functional_exact"] >= metrics["independent_modal_functional_exact"]
        and metrics["removed_abstain"] == 48
        and metrics["control_functional_exact"] / 48 <= 0.10
    )
    if (
        stored.get("metrics") != metrics
        or stored.get("verdict") != ("PASS" if passed else "FAIL")
        or stored.get("source_rows_sha256") != sha256_file(source_rows)
        or stored.get("package", {}).get("sha256") != sha256_file(package_path)
        or stored.get("package", {}).get("bytes") != package_path.stat().st_size
        or stored.get("extraction_result_sha256")
        != sha256_file(directory / "extraction/result.json")
        or (require_pass and not passed)
    ):
        raise R14Error("R19 development scientific gate failed")
    return metrics


def verify(
    r17_source: Path,
    r18_config: Path,
    r18_reveal: Path,
    r18_source: Path,
    run_dir: Path,
    *,
    _preverified_sources: bool = False,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    if not _preverified_sources:
        r17_strict = verify_r17_source(r17_source, "v2")
        if r17_strict != json_object(r17_source / "strict_source.json"):
            raise R14Error("R19 R17 source changed")
        r18_strict = verify_r18_source(r18_config, r18_reveal, r18_source)
        if r18_strict != json_object(r18_source / "strict_verification.json"):
            raise R14Error("R19 R18 source changed")
    receipt = _verified(run_dir / "receipt.json", "receipt")
    if (
        receipt.get("format") != "abi-r19-disclosed-development-qualification/1"
        or receipt.get("verdict") != "PASS"
        or receipt.get("protocol_sha256")
        != sha256_file(root / "experiments/contrastive_realization_r19/PUBLIC_PROTOCOL_V4.md")
        or receipt.get("source_training_steps") != 0
        or receipt.get("host_training_steps") != 0
        or receipt.get("teacher_present_at_compilation") is not False
        or receipt.get("teacher_present_at_package_execution") is not False
        or receipt.get("physical_primary_extractions") != 2
        or receipt.get("physical_control_attempts") != 2
        or receipt.get("full_abi_moonshot") != "OPEN"
    ):
        raise R14Error("R19 development receipt changed")
    expected_names = {"r17_public_v2", "r18_failed_hidden"}
    if set(receipt.get("datasets", {})) != expected_names:
        raise R14Error("R19 development dataset inventory changed")
    metrics = {
        "r17_public_v2": _verify_dataset(
            root,
            run_dir / "r17_public_v2",
            r17_source / "source_observations.jsonl",
            19_001,
            receipt["datasets"]["r17_public_v2"],
        ),
        "r18_failed_hidden": _verify_dataset(
            root,
            run_dir / "r18_failed_hidden",
            r18_source / "source_observations.jsonl",
            19_002,
            receipt["datasets"]["r18_failed_hidden"],
        ),
    }
    result = {
        "format": "abi-r19-development-strict-verification/1",
        "verdict": "PASS",
        "datasets_verified": 2,
        "evaluation_rows_verified": 96,
        "package_functional_exact": sum(
            item["package_functional_exact"] for item in metrics.values()
        ),
        "control_functional_exact": sum(
            item["control_functional_exact"] for item in metrics.values()
        ),
        "source_exact_regressions": sum(
            item["source_exact_regressions"] for item in metrics.values()
        ),
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r17-source", type=Path, required=True)
    parser.add_argument("--r18-config", type=Path, required=True)
    parser.add_argument("--r18-reveal", type=Path, required=True)
    parser.add_argument("--r18-source", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(
        args.r17_source,
        args.r18_config,
        args.r18_reveal,
        args.r18_source,
        args.run_dir,
    )
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
