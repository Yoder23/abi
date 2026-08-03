"""Import a pinned local open-weight source snapshot into ABI evidence.

The importer is deliberately byte-oriented.  It does not load or transform the
model and it accepts only an explicit allowlist with exact hashes and sizes.
The resulting source package is immutable and remains a source, never a cake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence


FORMAT = "abi-pinned-open-weight-source-snapshot/1"


class SourceSnapshotImportError(RuntimeError):
    """Raised when a pinned source snapshot differs from its preregistration."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def import_snapshot(
    *,
    source_path: str | Path,
    output_path: str | Path,
    model_id: str,
    revision: str,
    expected_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SourceSnapshotImportError(f"source package is immutable: {output_path}")
    if not model_id or not revision or not expected_files:
        raise SourceSnapshotImportError("source identity or file lock is missing")
    allowed = set(expected_files)
    actual = {path.name for path in source_path.iterdir() if path.is_file()}
    if actual != allowed:
        raise SourceSnapshotImportError(
            f"source file inventory changed: expected={sorted(allowed)} actual={sorted(actual)}"
        )
    verified: list[dict[str, Any]] = []
    for name in sorted(expected_files):
        source = source_path / name
        lock = expected_files[name]
        size = source.stat().st_size
        digest = _sha256_file(source)
        if size != int(lock.get("bytes", -1)) or digest != lock.get("sha256"):
            raise SourceSnapshotImportError(f"pinned source file changed: {name}")
        verified.append({"path": name, "bytes": size, "sha256": digest})
    output_path.mkdir(parents=True, exist_ok=False)
    for row in verified:
        shutil.copyfile(source_path / row["path"], output_path / row["path"])
    for row in verified:
        copied = output_path / row["path"]
        if copied.stat().st_size != row["bytes"] or _sha256_file(copied) != row["sha256"]:
            raise SourceSnapshotImportError(f"source copy changed: {row['path']}")
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "status": "PINNED_SOURCE_SNAPSHOT_NOT_DEPLOYABLE_CAKE",
        "model_id": model_id,
        "revision": revision,
        "source_path_at_import": str(source_path),
        "files": verified,
        "total_bytes": sum(row["bytes"] for row in verified),
        "transformations": 0,
        "teacher_present_at_layercake_inference": False,
        "claim_boundary": (
            "This package is a byte-exact local copy of a pinned open-weight "
            "source snapshot for ABI extraction. It is not a LayerCake cake, "
            "an English transfer result, or a deployable runtime."
        ),
    }
    manifest["manifest_sha256"] = _canonical_sha(manifest)
    (output_path / "abi_source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file-lock", required=True)
    args = parser.parse_args(argv)
    lock = json.loads(Path(args.file_lock).read_text(encoding="utf-8"))
    result = import_snapshot(
        source_path=args.source,
        output_path=args.output,
        model_id=args.model_id,
        revision=args.revision,
        expected_files=lock["files"],
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "manifest_sha256": result["manifest_sha256"],
                "total_bytes": result["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
