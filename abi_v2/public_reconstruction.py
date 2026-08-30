"""Reconstruct and verify ABI solely from a hash-addressed public manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .build_final_validation_bundle import PREFIX, verify_archive
from .final_validation import evidence_hash
from .public_release import verify_manifest

FORBIDDEN_DIRECTORIES = {".git", ".pytest_cache", ".venv", "__pycache__", "build", "dist"}


class PublicReconstructionError(RuntimeError):
    """Raised when public bytes cannot be reconstructed exactly."""


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicReconstructionError(f"manifest unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise PublicReconstructionError(f"manifest must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_assets(manifest: Path, destination: Path) -> list[dict[str, Any]]:
    value = _object(manifest)
    destination = destination.resolve()
    if destination.exists():
        raise PublicReconstructionError(f"fresh asset directory already exists: {destination}")
    destination.mkdir(parents=True)
    rows = []
    for asset in value.get("assets", []):
        name = str(asset.get("name", ""))
        if not name or Path(name).name != name:
            raise PublicReconstructionError("unsafe public asset name")
        target = destination / name
        partial = destination / f"{name}.partial"
        url = str(asset["download_url"])
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != "github.com":
            raise PublicReconstructionError(f"non-GitHub public asset URL: {name}")
        request = urllib.request.Request(
            url, headers={"User-Agent": "abi-public-reconstruction/1"}
        )
        with urllib.request.urlopen(request, timeout=120) as incoming, partial.open("xb") as output:
            while block := incoming.read(8 * 1024 * 1024):
                output.write(block)
        digest = _sha256(partial)
        if partial.stat().st_size != int(asset["bytes"]) or digest != asset["sha256"]:
            raise PublicReconstructionError(f"downloaded public asset changed: {name}")
        partial.replace(target)
        rows.append({"name": name, "bytes": target.stat().st_size, "sha256": digest})
    verify_manifest(manifest, destination)
    return rows


def extract_archive(archive: Path, destination: Path) -> Path:
    archive, destination = archive.resolve(), destination.resolve()
    if destination.exists():
        raise PublicReconstructionError(
            f"fresh reconstruction directory already exists: {destination}"
        )
    verification = verify_archive(archive)
    if verification["status"] != "PASS_EXACT_ARCHIVE_IDENTITY":
        raise PublicReconstructionError("public archive failed exact identity verification")
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        if len(names) != len(set(names)):
            raise PublicReconstructionError("public archive contains duplicate members")
        for info in handle.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise PublicReconstructionError(f"unsafe archive member: {info.filename}")
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise PublicReconstructionError(f"archive member escaped root: {info.filename}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with handle.open(info) as incoming, target.open("xb") as output:
                while block := incoming.read(8 * 1024 * 1024):
                    output.write(block)
    root = destination / PREFIX / "abi_release"
    if not root.is_dir():
        raise PublicReconstructionError("archive lacks the ABI release root")
    return root


def _run(root: Path, command: list[str]) -> dict[str, Any]:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(root)}
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _git(clone: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=clone, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise PublicReconstructionError(completed.stderr.strip() or "git verification failed")
    return completed.stdout.strip()


def reconstruct(manifest: Path, tag_clone: Path, workspace: Path) -> dict[str, Any]:
    manifest, tag_clone, workspace = (
        manifest.resolve(),
        tag_clone.resolve(),
        workspace.resolve(),
    )
    if workspace.exists():
        raise PublicReconstructionError(f"fresh public workspace already exists: {workspace}")
    started = time.perf_counter()
    manifest_value = _object(manifest)
    if not (tag_clone / ".git").exists():
        raise PublicReconstructionError("tag clone lacks independent Git metadata")
    if _git(tag_clone, "status", "--short"):
        raise PublicReconstructionError("tag clone is not clean")
    if _git(tag_clone, "rev-parse", "HEAD") != manifest_value.get("release_commit"):
        raise PublicReconstructionError("tag-clone HEAD differs from public manifest")
    if _git(tag_clone, "rev-list", "-n", "1", str(manifest_value.get("release_tag"))) != (
        manifest_value.get("release_commit")
    ):
        raise PublicReconstructionError("public release tag does not resolve to release commit")
    assets = download_assets(manifest, workspace / "assets")
    archive_asset = next(
        row
        for row in manifest_value["assets"]
        if row["role"] == "definitive_repaired_validation_archive"
    )
    archive = workspace / "assets" / archive_asset["name"]
    root = extract_archive(archive, workspace / "reconstructed")
    forbidden = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir() and path.name in FORBIDDEN_DIRECTORIES
    )
    if forbidden:
        raise PublicReconstructionError(f"development directories entered archive: {forbidden}")
    strict = _run(
        root,
        [
            sys.executable,
            "-B",
            "-m",
            "abi_v2.strict_validation",
            "--root",
            ".",
        ],
    )
    expected_strict_sha256 = str(
        manifest_value.get("strict_certificate", {}).get("sha256", "")
    )
    strict_identity_exact = (
        strict["exit_code"] == 0
        and len(expected_strict_sha256) == 64
        and strict["stdout_sha256"] == expected_strict_sha256
    )
    tests = _run(
        root,
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_abi_v2_strict_validation.py",
            "tests/test_abi_v2_final_validation.py",
            "tests/test_abi_v2_external_bundle.py",
        ],
    )
    passed = strict_identity_exact and tests["exit_code"] == 0
    value = {
        "format": "abi-v2-public-reconstruction/1",
        "status": (
            "PASS_PUBLIC_MANIFEST_ONLY_RECONSTRUCTION"
            if passed
            else "FAIL_PUBLIC_RECONSTRUCTION"
        ),
        "release_tag": manifest_value["release_tag"],
        "release_commit": manifest_value["release_commit"],
        "tag_clone_clean": True,
        "tag_clone_head": _git(tag_clone, "rev-parse", "HEAD"),
        "manifest_sha256": _sha256(manifest),
        "archive_sha256": _sha256(archive),
        "assets": assets,
        "reconstructed_root": root.relative_to(workspace).as_posix(),
        "development_directories_present": forbidden,
        "strict_raw_recomputation": strict,
        "strict_certificate_identity_exact": strict_identity_exact,
        "tests": tests,
        "wall_seconds": time.perf_counter() - started,
    }
    value["evidence_sha256"] = evidence_hash(value)
    return value


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tag-clone", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise PublicReconstructionError(f"immutable receipt exists: {output}")
    value = reconstruct(
        Path(args.manifest), Path(args.tag_clone), Path(args.workspace)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
