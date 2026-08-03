"""Build a deterministic manifest for large local ABI research catalogs.

The files remain on the research host and out of ordinary Git history. The
manifest makes that retention explicit and binds every byte by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


FORMAT = "abi-local-research-artifacts-manifest/1"
DEFAULT_THRESHOLD_BYTES = 1_048_576


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_large_catalogs(
    repository_root: Path,
    *,
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
    tracked_paths: set[str] | None = None,
) -> list[dict[str, object]]:
    if threshold_bytes <= 0:
        raise ValueError("threshold_bytes must be positive")
    root = repository_root.resolve()
    catalog_root = root / "catalogs"
    if not catalog_root.is_dir():
        raise FileNotFoundError(f"catalog directory does not exist: {catalog_root}")

    tracked = tracked_paths or set()
    artifacts: list[dict[str, object]] = []
    for path in sorted(catalog_root.rglob("*.json")):
        if not path.is_file() or path.stat().st_size < threshold_bytes:
            continue
        relative_path = path.relative_to(root).as_posix()
        artifacts.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "classification": "historical_generated_catalog",
                "git_state": "TRACKED" if relative_path in tracked else "LOCAL_ONLY",
                "retention": "PRESERVED_AND_CONTENT_ADDRESSED",
            }
        )
    return artifacts


def git_tracked_paths(repository_root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={repository_root.as_posix()}", "ls-files", "--", "catalogs"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return set()
    return {line.strip().replace("\\", "/") for line in completed.stdout.splitlines() if line.strip()}


def build_manifest(
    repository_root: Path,
    *,
    status_date: str,
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
    tracked_paths: set[str] | None = None,
) -> dict[str, object]:
    artifacts = inventory_large_catalogs(
        repository_root,
        threshold_bytes=threshold_bytes,
        tracked_paths=tracked_paths,
    )
    tracked_count = sum(item["git_state"] == "TRACKED" for item in artifacts)
    return {
        "format": FORMAT,
        "status_date": status_date,
        "status": "LOCAL_ARTIFACTS_PRESERVED_AND_CONTENT_ADDRESSED",
        "scope": "JSON catalogs at or above the declared byte threshold",
        "threshold_bytes": threshold_bytes,
        "artifact_count": len(artifacts),
        "total_bytes": sum(int(item["bytes"]) for item in artifacts),
        "tracked_artifact_count": tracked_count,
        "local_only_artifact_count": len(artifacts) - tracked_count,
        "git_policy": {
            "deletion_authorized": False,
            "new_large_generated_catalogs": "LOCAL_ONLY_AND_CONTENT_ADDRESSED",
            "previously_tracked_large_catalogs": "PRESERVED_WITHOUT_HISTORY_REWRITE",
            "protocols_generators_decisions_and_compact_manifests_tracked": True,
            "restore_requires_exact_sha256": True,
        },
        "artifacts": artifacts,
    }


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ABI_LOCAL_RESEARCH_ARTIFACTS_MANIFEST_V1.json"),
    )
    parser.add_argument("--status-date", required=True)
    parser.add_argument(
        "--threshold-bytes",
        type=int,
        default=DEFAULT_THRESHOLD_BYTES,
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = args.repository_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = repository_root / output
    manifest = build_manifest(
        repository_root,
        status_date=args.status_date,
        threshold_bytes=args.threshold_bytes,
        tracked_paths=git_tracked_paths(repository_root),
    )
    write_manifest(output, manifest)
    print(
        f"wrote {output} with {manifest['artifact_count']} artifacts "
        f"and {manifest['total_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
