"""Build and verify the turnkey ABI final-validation clean-room archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .final_validation import sha256_file, write_json

PREFIX = "abi-final-validation-v2"
FORBIDDEN = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".venv"}


class FinalBundleError(RuntimeError):
    """Raised when the curated final-validation archive is not reproducible."""


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str
    classification: str


def checklist() -> dict[str, Any]:
    return {
        "format": "abi-final-external-reproduction-checklist/1",
        "status": "CLOSED_UNTIL_PUBLIC_RECONSTRUCTION_AND_BLIND_RED_TEAM_PASS",
        "repository": "https://github.com/Yoder23/abi",
        "frozen_commit": "see results/abi_final_validation_v2/frozen_release_candidate_r4.json",
        "frozen_tag": "see results/abi_final_validation_v2/frozen_release_candidate_r4.json",
        "commands": [
            "abi-reproduce verify",
            "abi-reproduce certify-hosts",
            "abi-reproduce capability-matrix",
            "abi-reproduce causality",
            "abi-reproduce isolation",
            "abi-reproduce performance",
            "abi-reproduce hostile-audit",
            "abi-reproduce report",
        ],
        "operator_must_record": [
            "operator identity and signed attestation",
            "CPU, GPU, RAM, VRAM, OS",
            "Python and compiler versions",
            "package/runtime versions",
            "exact commands and exit codes",
            "all raw outputs and timings",
            "SHA-256 inventory of returned evidence",
        ],
        "hardware_rule": "must differ from the ABI development RTX 3080 Laptop GPU system",
        "excluded_payload_classes": [
            "host model weights",
            "development caches",
            "trainer checkpoints",
            "hidden expected outputs",
            "human rating results",
            "source teacher runtime",
        ],
        "disclosed_payload_classes": ["public locked evaluator/reference outputs"],
        "prerequisites": [
            "immutable public capability and archive publication",
            "fresh reconstruction from the public tag and manifests",
            "fresh blind Codex red-team against the repaired tag",
        ],
        "minimum_information_status": "PENDING_AFTER_EXTERNAL_VALIDATION",
    }


def environment_lock() -> dict[str, Any]:
    return {
        "format": "abi-final-external-environment-lock/1",
        "python": "3.10",
        "packages": {
            "cryptography": "46.0.5",
            "huggingface-hub": "0.36.0",
            "numpy": "2.2.6",
            "psutil": "7.0.0",
            "safetensors": "0.7.0",
            "tokenizers": "0.22.1",
            "torch": "2.7.1",
            "transformers": "4.57.3",
        },
        "torch_build_rule": "operator selects the matching CPU/CUDA build and records the exact local version",
        "compiler_rule": "record compiler and platform; exact development hardware is prohibited",
    }


def prepare(root: Path) -> dict[str, Any]:
    root = root.resolve()
    write_json(root / "external_reproduction/checklist.json", checklist())
    write_json(root / "external_reproduction/environment.lock.json", environment_lock())
    return {
        "status": "PASS_EXTERNAL_REPRODUCTION_METADATA_PREPARED",
        "checklist": "external_reproduction/checklist.json",
        "environment_lock": "external_reproduction/environment.lock.json",
    }


def _append(records: list[BundleFile], root: Path, relative: str, classification: str) -> None:
    source = (root / relative).resolve()
    if not source.is_file():
        raise FinalBundleError(f"missing bundle input: {relative}")
    if any(part in FORBIDDEN for part in Path(relative).parts):
        raise FinalBundleError(f"forbidden bundle path: {relative}")
    records.append(BundleFile(source, f"abi_release/{Path(relative).as_posix()}", classification))


def _tree(
    records: list[BundleFile], root: Path, relative: str, patterns: tuple[str, ...], classification: str
) -> None:
    base = root / relative
    for pattern in patterns:
        for path in sorted(base.rglob(pattern)):
            if path.is_file() and not any(part in FORBIDDEN for part in path.parts):
                _append(records, root, path.relative_to(root).as_posix(), classification)


def collect(root: Path, layercake_root: Path) -> list[BundleFile]:
    root, layercake_root = root.resolve(), layercake_root.resolve()
    records: list[BundleFile] = []
    for relative in (
        "LICENSE",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "AGENTS.md",
        "ABI_CAPABILITY_COMPILER_PHASE7_PRODUCT_MANIFEST_V1.json",
        "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_PROTOCOL_V1040.json",
        "ABI_CAPABILITY_COMPILER_PHASE4_B40_V25_PRODUCT_CONFORMANCE_PROTOCOL_V960.json",
        "catalogs/capability_compiler_phase1_frozen_v1.json",
        "evidence/current/segregation/ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE.json",
        "evidence/current/segregation/ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V2.json",
        "evidence/current/segregation/ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V3.json",
        "evidence/current/segregation/ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V4.json",
        "evidence/current/segregation/ABI_CORE_DOMAIN_SEGREGATION_IMPLEMENTATION_CERTIFICATE_V5.json",
        "evidence/current/segregation/ABI_ENGLISH_CORE_DOMAIN_SEGREGATION_CONTRACT_V2.json",
        "evidence/current/segregation/domain_ontology_v1.json",
        "evidence/current/segregation/english_and_first_domains_certification_v6.json",
        "evidence/current/segregation/intrinsic_english_search_validation_v83.json",
        "results/abi_capability_compiler_phase4_clarification_route_replication/B40-seed104729-v927/evaluation/development_outputs.jsonl",
        "results/abi_capability_compiler_phase6_composition/run_v1032/seed104729/observations.jsonl",
        "results/abi_capability_compiler_phase7_integrated/materialized_v1052/phase7-final-english-core.cake",
        "results/abi_moonshot/packages/abi-python-token-plan-seed9824.cake",
        "results/abi_moonshot/packages/abi-chemistry-token-plan-seed9824.cake",
        "results/abi_moonshot/packages/abi-civics-token-plan-seed9824.cake",
        "results/abi_moonshot/packages/abi-python-token-plan-seed9824.pub",
        "results/abi_moonshot/packages/abi-chemistry-token-plan-seed9824.pub",
        "results/abi_moonshot/packages/abi-civics-token-plan-seed9824.pub",
    ):
        _append(records, root, relative, "frozen_runtime_or_evaluation_input")
    _tree(records, root, "abi", ("*.py",), "abi_source")
    _tree(
        records,
        root,
        "abi_v2",
        ("*.py", "*.json", "*.sh"),
        "final_validation_source_or_spec",
    )
    _tree(
        records,
        root,
        "tests",
        (
            "test_public_release.py",
            "test_capability_pipeline.py",
            "test_capability_segregation.py",
            "test_abi_v2_*.py",
            "test_human_rate.py",
        ),
        "supported_release_tests",
    )
    _tree(records, root, "results/abi_v2", ("*.json", "*.jsonl", "*.md"), "locked_raw_evidence")
    _tree(
        records,
        root,
        "results/abi_host_independence",
        ("*.json", "*.jsonl", "*.md"),
        "bound_predecessor_release_evidence",
    )
    _tree(records, root, "results/abi_final_validation", ("*.json", "*.md"), "final_validation_evidence")
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_validation_r4_content_bound.json",
        "repaired_strict_certificate",
    )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/frozen_release_candidate_r4.json",
        "repaired_frozen_candidate",
    )
    if (root / "results/abi_final_validation_v2/strict_hostile_pre_public_r4.json").is_file():
        _append(
            records,
            root,
            "results/abi_final_validation_v2/strict_hostile_pre_public_r4.json",
            "repaired_strict_hostile_receipt",
        )
    for relative in (
        "results/abi_final_validation_v2/isolated_certification_strict",
        "results/abi_final_validation_v2/isolated_certification_strict_r4_content_bound",
        "results/abi_final_validation_v2/live_causality",
        "results/abi_final_validation_v2/live_isolation",
    ):
        _tree(
            records,
            root,
            relative,
            ("*.json", "*.jsonl", "*.md", "*.txt"),
            "repaired_raw_validation_evidence",
        )
    for relative in (
        "results/abi_final_validation_v2/isolated_certification",
        "results/abi_final_validation_v2/live_causality_untrusted_boolean_history",
        "results/abi_final_validation_v2/pre_strict_capsule_root_history",
        "results/abi_final_validation_v2/isolated_certification_strict_r2_stale_source_history",
        "results/abi_final_validation_v2/live_causality_r2_logged_state_history",
    ):
        _tree(
            records,
            root,
            relative,
            ("*.json", "*.jsonl", "*.md", "*.txt", "*.py"),
            "superseded_historical_evidence",
        )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_validation_pre_boolean_cleanup.json",
        "superseded_historical_evidence",
    )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_validation_pre_hostile_fail_closed_fix.json",
        "superseded_historical_evidence",
    )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_validation_pre_declarative_boolean_audit.json",
        "superseded_historical_evidence",
    )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_validation_pre_clean_archive_lineage_fix.json",
        "superseded_historical_evidence",
    )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_hostile_pre_declarative_boolean_audit.json",
        "superseded_historical_evidence",
    )
    _append(
        records,
        root,
        "results/abi_final_validation_v2/strict_hostile_pre_clean_archive_lineage_fix.json",
        "superseded_historical_evidence",
    )
    for relative in (
        "results/abi_final_validation_v2/strict_validation_r2_pre_physical_live_repair.json",
        "results/abi_final_validation_v2/frozen_release_candidate_r2_history.json",
        "results/abi_final_validation_v2/strict_hostile_pre_public_r2_history.json",
        "results/abi_final_validation_v2/strict_hostile_pre_source_binding_fix.json",
        "results/abi_final_validation_v2/strict_validation_pre_disposable_input_closure_fix.json",
        "results/abi_final_validation_v2/strict_validation_pre_transitive_input_closure_fix.json",
    ):
        _append(records, root, relative, "superseded_historical_evidence")
    _tree(records, root, "results/abi_capability_compiler_phase2/human_rating_packet_v1", ("*.json", "*.jsonl"), "sealed_human_packet")
    _tree(records, root, "docs", ("*.md", "*.json"), "documentation")
    _tree(records, root, "review_packet", ("*.md",), "review_packet")
    for path in sorted((root / "external_reproduction").glob("*")):
        if path.is_file() and path.suffix in {".md", ".json"}:
            _append(
                records,
                root,
                path.relative_to(root).as_posix(),
                "external_operator_material",
            )

    for base in ("layercake", "layercake_extensions"):
        for path in sorted((layercake_root / base).rglob("*.py")):
            relative = path.relative_to(layercake_root).as_posix()
            records.append(
                BundleFile(path.resolve(), f"layercake_release/{relative}", "layercake_runtime_source")
            )
    for relative in ("pyproject.toml", "moonshot/canonical_route_isolated_clarification_core_abi_v25.json"):
        path = layercake_root / relative
        records.append(
            BundleFile(path.resolve(), f"layercake_release/{relative}", "layercake_runtime_spec")
        )

    # The archive receipt binds the completed archive and therefore remains an
    # out-of-band tracked release record; including an older receipt would make
    # the archive recursively or incorrectly self-describing.
    receipt_member = "abi_release/results/abi_final_validation/external_archive_receipt.json"
    records = [record for record in records if record.archive_path != receipt_member]

    unique: dict[str, BundleFile] = {}
    for record in records:
        previous = unique.setdefault(record.archive_path, record)
        if previous.source != record.source:
            raise FinalBundleError(f"duplicate archive member: {record.archive_path}")
    return [unique[name] for name in sorted(unique)]


def build(root: Path, layercake_root: Path, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FinalBundleError(f"refusing to overwrite: {output}")
    files = collect(root, layercake_root)
    candidate = json.loads(
        (root / "results/abi_final_validation_v2/frozen_release_candidate_r4.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = {
        "format": "abi-final-validation-clean-room-manifest/2",
        "status": "REPAIRED_CANDIDATE_REQUIRES_PUBLIC_RECONSTRUCTION_AND_RED_TEAM",
        "frozen_technical_commit": candidate["technical_proof_commit"],
        "frozen_technical_tag": candidate["technical_proof_tag"],
        "excluded_payload_file_counts": {
            "host_model_weights": 0,
            "development_caches": 0,
            "hidden_expected_outputs": 0,
        },
        "public_reference_output_files": 2,
        "files": [
            {
                "path": record.archive_path,
                "bytes": record.source.stat().st_size,
                "sha256": sha256_file(record.source),
                "classification": record.classification,
            }
            for record in files
        ],
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = (2026, 8, 25, 0, 0, 0)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for record in files:
            info = zipfile.ZipInfo(f"{PREFIX}/{record.archive_path}", timestamp)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            with record.source.open("rb") as incoming, archive.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                    target.write(block)
        info = zipfile.ZipInfo(f"{PREFIX}/manifest.json", timestamp)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes)
    return {
        "format": "abi-final-validation-archive-receipt/2",
        "status": "PASS_REPAIRED_ARCHIVE_BUILT_REQUIRES_PUBLIC_RECONSTRUCTION",
        "archive": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": len(files),
        "next_required_actions": [
            "publish immutable archive and capability assets",
            "reconstruct from public manifests in a new directory",
            "run fresh blind red-team",
        ],
    }


def verify_archive(path: Path) -> dict[str, Any]:
    path = path.resolve()
    failures: list[str] = []
    with zipfile.ZipFile(path, "r") as archive:
        manifest_name = f"{PREFIX}/manifest.json"
        manifest = json.loads(archive.read(manifest_name))
        expected = {manifest_name}
        for row in manifest["files"]:
            name = f"{PREFIX}/{row['path']}"
            expected.add(name)
            try:
                payload = archive.read(name)
            except KeyError:
                failures.append(f"missing:{name}")
                continue
            if len(payload) != row["bytes"] or hashlib.sha256(payload).hexdigest() != row["sha256"]:
                failures.append(f"identity:{name}")
        unexpected = sorted(set(archive.namelist()) - expected)
        failures.extend(f"unexpected:{name}" for name in unexpected)
    return {
        "format": "abi-final-validation-archive-verification/2",
        "status": "PASS_EXACT_ARCHIVE_IDENTITY" if not failures else "FAIL_ARCHIVE_IDENTITY",
        "archive_sha256": sha256_file(path),
        "files_verified": len(expected) - 1 - len([x for x in failures if x.startswith("missing:")]),
        "failures": failures,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--layercake-root", default="../layercake_release")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--receipt")
    parser.add_argument("--verify")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.prepare:
        result = prepare(root)
    elif args.verify:
        result = verify_archive(Path(args.verify))
    elif args.output:
        result = build(root, root / args.layercake_root, Path(args.output))
    else:
        parser.error("choose --prepare, --output, or --verify")
    if args.receipt:
        write_json((root / args.receipt).resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith(("PASS", "READY")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
