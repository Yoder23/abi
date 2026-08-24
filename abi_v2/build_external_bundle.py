"""Build the curated ABI V2 different-hardware reproduction archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ExternalBundleBuildError(RuntimeError):
    """Raised when a clean-room archive cannot be built safely."""


@dataclass(frozen=True)
class BundleFile:
    source: Path
    archive_path: str
    classification: str


ROOT_FILES = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "ABI_CAPABILITY_COMPILER_PHASE7_PRODUCT_MANIFEST_V1.json",
    "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_PROTOCOL_V1040.json",
    "ABI_CAPABILITY_COMPILER_PHASE4_B40_V25_PRODUCT_CONFORMANCE_PROTOCOL_V960.json",
)

PUBLIC_EVALUATION_FILES = (
    "catalogs/capability_compiler_phase1_frozen_v1.json",
    "evidence/current/segregation/english_and_first_domains_certification_v6.json",
    "results/abi_capability_compiler_phase4_clarification_route_replication/"
    "B40-seed104729-v927/evaluation/development_outputs.jsonl",
    "results/abi_capability_compiler_phase6_composition/"
    "run_v1032/seed104729/observations.jsonl",
)

CAPABILITY_FILES = (
    "results/abi_capability_compiler_phase7_integrated/"
    "materialized_v1052/phase7-final-english-core.cake",
    "results/abi_moonshot/packages/abi-python-token-plan-seed9824.cake",
    "results/abi_moonshot/packages/abi-chemistry-token-plan-seed9824.cake",
    "results/abi_moonshot/packages/abi-civics-token-plan-seed9824.cake",
)

PUBLIC_TRUST_FILES = (
    "results/abi_moonshot/packages/abi-python-token-plan-seed9824.pub",
    "results/abi_moonshot/packages/abi-chemistry-token-plan-seed9824.pub",
    "results/abi_moonshot/packages/abi-civics-token-plan-seed9824.pub",
)

LAYERCAKE_FILES = (
    "moonshot/canonical_route_isolated_clarification_core_abi_v25.json",
)

FORBIDDEN_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "dist"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _append(
    records: list[BundleFile],
    *,
    source: Path,
    archive_path: str,
    classification: str,
) -> None:
    if not source.is_file():
        raise ExternalBundleBuildError(f"required bundle input is missing: {source}")
    normalized = Path(archive_path)
    if any(part in FORBIDDEN_PARTS for part in normalized.parts):
        raise ExternalBundleBuildError(f"forbidden archive path: {archive_path}")
    records.append(BundleFile(source.resolve(), normalized.as_posix(), classification))


def _append_python_tree(
    records: list[BundleFile],
    *,
    source_root: Path,
    archive_root: str,
    classification: str,
) -> None:
    for path in sorted(source_root.rglob("*.py")):
        if any(part in FORBIDDEN_PARTS for part in path.parts):
            continue
        relative = path.relative_to(source_root).as_posix()
        _append(
            records,
            source=path,
            archive_path=f"{archive_root}/{relative}",
            classification=classification,
        )


def collect_bundle_files(root: Path, layercake_root: Path) -> list[BundleFile]:
    root, layercake_root = root.resolve(), layercake_root.resolve()
    records: list[BundleFile] = []
    for relative in ROOT_FILES:
        _append(
            records,
            source=root / relative,
            archive_path=f"abi_release/{relative}",
            classification="source_or_protocol",
        )
    _append_python_tree(
        records,
        source_root=root / "abi",
        archive_root="abi_release/abi",
        classification="abi_runtime_source",
    )
    _append_python_tree(
        records,
        source_root=root / "abi_v2",
        archive_root="abi_release/abi_v2",
        classification="abi_v2_source",
    )
    for path in sorted((root / "abi_v2").rglob("*.json")):
        _append(
            records,
            source=path,
            archive_path=f"abi_release/{path.relative_to(root).as_posix()}",
            classification="abi_v2_specification",
        )
    for path in sorted((root / "tests").glob("test_abi_v2_*.py")):
        _append(
            records,
            source=path,
            archive_path=f"abi_release/tests/{path.name}",
            classification="test_suite",
        )
    for path in sorted((root / "results/abi_v2").rglob("*")):
        if not path.is_file() or path.suffix == ".zip":
            continue
        _append(
            records,
            source=path,
            archive_path=f"abi_release/{path.relative_to(root).as_posix()}",
            classification="local_reference_evidence",
        )
    for relative in PUBLIC_EVALUATION_FILES:
        classification = (
            "public_reference_output" if relative.endswith(".jsonl") else "public_evaluation_suite"
        )
        _append(
            records,
            source=root / relative,
            archive_path=f"abi_release/{relative}",
            classification=classification,
        )
    for relative in CAPABILITY_FILES:
        _append(
            records,
            source=root / relative,
            archive_path=f"abi_release/{relative}",
            classification="immutable_capability_package",
        )
    for relative in PUBLIC_TRUST_FILES:
        _append(
            records,
            source=root / relative,
            archive_path=f"abi_release/{relative}",
            classification="public_research_trust_material",
        )
    _append_python_tree(
        records,
        source_root=layercake_root / "layercake",
        archive_root="layercake_release/layercake",
        classification="layercake_runtime_source",
    )
    _append_python_tree(
        records,
        source_root=layercake_root / "layercake_extensions",
        archive_root="layercake_release/layercake_extensions",
        classification="layercake_runtime_source",
    )
    for relative in LAYERCAKE_FILES:
        _append(
            records,
            source=layercake_root / relative,
            archive_path=f"layercake_release/{relative}",
            classification="layercake_runtime_specification",
        )

    by_name: dict[str, BundleFile] = {}
    for record in records:
        previous = by_name.setdefault(record.archive_path, record)
        if previous.source != record.source:
            raise ExternalBundleBuildError(f"duplicate archive member: {record.archive_path}")
    return [by_name[name] for name in sorted(by_name)]


def _model_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    hosts = protocol["host_registry"]
    return {
        key: {
            field: value
            for field, value in hosts[key].items()
            if field
            in {
                "host_id",
                "architecture",
                "model",
                "revision",
                "snapshot_inventory_sha256",
                "checkpoint_sha256",
                "tokenizer_sha256",
            }
        }
        for key in ("layercake", "qwen2", "pythia")
    }


def build_bundle(
    root: Path,
    *,
    layercake_root: Path,
    output: Path,
    require_clean: bool = True,
) -> dict[str, Any]:
    root, layercake_root, output = root.resolve(), layercake_root.resolve(), output.resolve()
    if output.exists():
        raise ExternalBundleBuildError(f"refusing to overwrite archive: {output}")
    if require_clean and _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ExternalBundleBuildError("tracked worktree changes must be committed before bundling")
    commit = _git(root, "rev-parse", "HEAD")
    amendment = json.loads(
        (root / "abi_v2/matrix_protocol_amendment3.json").read_text("utf-8")
    )
    base_protocol = json.loads((root / amendment["base_protocol"]).read_text("utf-8"))
    protocol = {
        **base_protocol,
        **amendment,
        "bindings": {**base_protocol["bindings"], **amendment.get("bindings", {})},
    }
    files = collect_bundle_files(root, layercake_root)
    manifest_files = [
        {
            "path": record.archive_path,
            "bytes": record.source.stat().st_size,
            "sha256": _sha256(record.source),
            "classification": record.classification,
        }
        for record in files
    ]
    manifest = {
        "format": "abi-v2-clean-room-manifest/1",
        "status": "READY_FOR_INDEPENDENT_DIFFERENT_HARDWARE_EXECUTION",
        "abi_repository": "https://github.com/Yoder23/abi",
        "abi_commit": commit,
        "layercake_repository": "https://github.com/Yoder23/layercake",
        "layercake_commit_recorded_by_product_manifest": "a87a653dbdb1a4e5f713baf7bc508d508277e00d",
        "matrix_protocol_sha256": _sha256(root / "abi_v2/matrix_protocol_amendment3.json"),
        "models": _model_manifest(protocol),
        "capability_packages": protocol["capability_packages"],
        "host_adapters": json.loads(
            (root / "results/abi_v2/adapters/manifest.json").read_text("utf-8")
        )["adapters"],
        "runtime_manifest": "abi_release/abi_v2/external_runtime_manifest.json",
        "raw_evidence_schema": (
            "abi_release/results/abi_v2/external_reproduction/raw_evidence.schema.json"
        ),
        "commands": "abi_release/results/abi_v2/external_reproduction/README.md",
        "evaluation_disclosure": {
            "evaluation_suites_included": True,
            "reference_outputs_included_and_public": True,
            "hidden_expected_outputs_included": False,
            "blind_holdout_claimed": False,
        },
        "model_weights_bundled": False,
        "model_weight_note": "Download the two exact open-weight snapshots at their pinned revisions, then verify every recorded hash before execution.",
        "development_caches_included": False,
        "independent_reproduction_claimed": False,
        "files": manifest_files,
    }
    manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    fixed_time = (2026, 8, 24, 0, 0, 0)
    prefix = "abi-v2-clean-room"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for record in files:
            info = zipfile.ZipInfo(f"{prefix}/{record.archive_path}", date_time=fixed_time)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            with record.source.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    target.write(block)
        info = zipfile.ZipInfo(f"{prefix}/manifest.json", date_time=fixed_time)
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_bytes)
    return {
        "format": "abi-v2-clean-room-archive-receipt/1",
        "status": "READY_FOR_EXTERNAL_OPERATOR",
        "archive": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": _sha256(output),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": len(files),
        "abi_commit": commit,
        "independent_execution_completed": False,
        "claim_boundary": "Archive construction and local identity verification do not satisfy independent different-hardware reproduction.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--layercake-root", default="../layercake_release")
    parser.add_argument("--output")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    output = (
        Path(args.output).resolve()
        if args.output
        else root
        / "results/abi_v2/external_reproduction"
        / f"abi-v2-clean-room-{commit[:12]}.zip"
    )
    receipt = build_bundle(
        root,
        layercake_root=(root / args.layercake_root).resolve(),
        output=output,
        require_clean=not args.allow_dirty,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
