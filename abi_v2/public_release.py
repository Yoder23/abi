"""Build or verify the hash-addressed ABI V2 public release asset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from .final_validation import CAPABILITY_PATHS

TAG_RE = re.compile(r"^[A-Za-z0-9._-]+$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REPOSITORY_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class PublicReleaseError(RuntimeError):
    """Raised when a public release cannot be bound or reproduced exactly."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise PublicReleaseError(f"required release asset missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PublicReleaseError(f"required release asset unreadable: {path}") from exc
    return digest.hexdigest()


def _asset(path: Path, *, role: str, repository: str, tag: str) -> dict[str, Any]:
    digest = sha256_file(path)
    name = path.name
    return {
        "name": name,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "content_address": f"sha256:{digest}",
        "download_url": (
            f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/"
            f"{quote(name, safe='')}"
        ),
    }


def build_manifest(
    *,
    root: Path,
    archive: Path,
    repository: str,
    tag: str,
    commit: str,
) -> dict[str, Any]:
    root, archive = root.resolve(), archive.resolve()
    if not REPOSITORY_RE.fullmatch(repository):
        raise PublicReleaseError(f"invalid GitHub repository: {repository}")
    if not TAG_RE.fullmatch(tag):
        raise PublicReleaseError(f"invalid release tag: {tag}")
    if not COMMIT_RE.fullmatch(commit):
        raise PublicReleaseError(f"invalid release commit: {commit}")
    assets = [
        _asset(archive, role="definitive_repaired_validation_archive", repository=repository, tag=tag)
    ]
    for capability, relative in sorted(CAPABILITY_PATHS.items()):
        assets.append(
            _asset(
                root / relative,
                role=f"immutable_{capability}_capability_package",
                repository=repository,
                tag=tag,
            )
        )
    strict_path = root / (
        "results/abi_final_validation_v2/strict_validation_r4_content_bound.json"
    )
    hostile_path = root / (
        "results/abi_final_validation_v2/strict_hostile_pre_public_r4.json"
    )
    return {
        "format": "abi-v2-public-release-assets/1",
        "repository": repository,
        "release_tag": tag,
        "release_commit": commit,
        "release_page": f"https://github.com/{repository}/releases/tag/{quote(tag, safe='')}",
        "assets": assets,
        "strict_certificate": {
            "path_in_archive": (
                "abi_release/results/abi_final_validation_v2/"
                "strict_validation_r4_content_bound.json"
            ),
            "sha256": sha256_file(strict_path),
        },
        "pre_public_hostile_receipt": {
            "path_in_archive": (
                "abi_release/results/abi_final_validation_v2/strict_hostile_pre_public_r4.json"
            ),
            "sha256": sha256_file(hostile_path),
        },
        "next_required_action": (
            "fresh clone and reconstruction from this manifest, followed by blind red-team"
        ),
    }


def verify_manifest(manifest_path: Path, asset_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicReleaseError(f"release manifest unavailable: {manifest_path}") from exc
    if not isinstance(value, dict) or value.get("format") != "abi-v2-public-release-assets/1":
        raise PublicReleaseError("release manifest format changed")
    repository = str(value.get("repository", ""))
    tag = str(value.get("release_tag", ""))
    commit = str(value.get("release_commit", ""))
    if (
        not REPOSITORY_RE.fullmatch(repository)
        or not TAG_RE.fullmatch(tag)
        or not COMMIT_RE.fullmatch(commit)
    ):
        raise PublicReleaseError("release identity binding missing or malformed")
    assets = value.get("assets")
    if not isinstance(assets, list) or len(assets) != 5:
        raise PublicReleaseError("release manifest must bind one archive and four capabilities")
    names = [str(row.get("name")) for row in assets]
    if len(set(names)) != len(names):
        raise PublicReleaseError("release manifest contains duplicate asset names")
    roles = {str(row.get("role")) for row in assets}
    expected_roles = {
        "definitive_repaired_validation_archive",
        *(f"immutable_{capability}_capability_package" for capability in CAPABILITY_PATHS),
    }
    if roles != expected_roles:
        raise PublicReleaseError("release manifest asset roles changed")
    verified = []
    for row in assets:
        path = asset_dir.resolve() / str(row["name"])
        digest = sha256_file(path)
        if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
            raise PublicReleaseError(f"public asset identity mismatch: {path.name}")
        if row.get("content_address") != f"sha256:{digest}":
            raise PublicReleaseError(f"public content address mismatch: {path.name}")
        expected_url = (
            f"https://github.com/{repository}/releases/download/{quote(tag, safe='')}/"
            f"{quote(path.name, safe='')}"
        )
        if row.get("download_url") != expected_url:
            raise PublicReleaseError(f"public download URL changed: {path.name}")
        verified.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
    return {
        "format": "abi-v2-public-release-assets-verification/1",
        "status": "PASS_EXACT_PUBLIC_ASSET_IDENTITY",
        "assets_verified": verified,
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise PublicReleaseError(f"refusing to overwrite immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--archive")
    parser.add_argument("--repository", default="Yoder23/abi")
    parser.add_argument("--tag")
    parser.add_argument("--commit")
    parser.add_argument("--output")
    parser.add_argument("--verify-manifest")
    parser.add_argument("--asset-dir")
    args = parser.parse_args(argv)
    if args.verify_manifest:
        if not args.asset_dir:
            parser.error("--verify-manifest requires --asset-dir")
        result = verify_manifest(Path(args.verify_manifest), Path(args.asset_dir))
    else:
        if not all((args.archive, args.tag, args.commit, args.output)):
            parser.error("build requires --archive, --tag, --commit, and --output")
        result = build_manifest(
            root=Path(args.root),
            archive=Path(args.archive),
            repository=args.repository,
            tag=args.tag,
            commit=args.commit,
        )
        _write_once(Path(args.output), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
