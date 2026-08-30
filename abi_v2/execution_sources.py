"""Canonical content binding for every transitive runtime source file."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, sha256_bytes


class ExecutionSourceError(RuntimeError):
    """Raised when the frozen execution source tree is incomplete."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def execution_source_manifest(root: Path) -> dict[str, Any]:
    """Hash all ABI and LayerCake code reachable by validation executions."""

    root = root.resolve()
    layercake_root = (root.parent / "layercake_release").resolve()
    trees = (
        ("abi", root / "abi", {".py"}),
        ("abi_v2", root / "abi_v2", {".py", ".json", ".sh"}),
        ("@layercake_release/layercake", layercake_root / "layercake", {".py"}),
        (
            "@layercake_release/layercake_extensions",
            layercake_root / "layercake_extensions",
            {".py"},
        ),
    )
    files: list[dict[str, Any]] = []
    for label, tree, suffixes in trees:
        if not tree.is_dir():
            raise ExecutionSourceError(f"execution source tree missing: {tree}")
        for path in sorted(tree.rglob("*"), key=lambda value: value.as_posix()):
            if (
                path.is_file()
                and path.suffix.casefold() in suffixes
                and "__pycache__" not in path.parts
            ):
                files.append(
                    {
                        "path": f"{label}/{path.relative_to(tree).as_posix()}",
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
    if not files or len({row["path"] for row in files}) != len(files):
        raise ExecutionSourceError("execution source inventory is empty or duplicated")
    return {
        "format": "abi-v2-transitive-execution-sources/1",
        "file_count": len(files),
        "files": files,
        "aggregate_sha256": sha256_bytes(canonical_json_bytes(files)),
    }
