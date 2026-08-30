"""Inventory and remove superseded generated result binaries safely.

The cleanup is intentionally narrow: untracked ONNX/safetensors files below
``results/`` and two explicitly named, reproducible ZIP bundles.  Every target
is hashed before removal and recorded in an immutable manifest.  Compact
scientific evidence and deployable ``.cake`` packages are never selected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REMOVABLE_SUFFIXES = {".onnx", ".safetensors"}
REMOVABLE_ZIPS = {
    "results/abi_final_mile/abi-final-mile-cleanroom.zip",
    "results/abi_v2/external_reproduction/abi-v2-clean-room-151cf8c59b91.zip",
}
CONFIRMATION = "REMOVE_HASHED_GENERATED_RESULT_BINARIES"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "results"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def targets(root: Path) -> list[Path]:
    results_root = (root / "results").resolve()
    selected: list[Path] = []
    for path in results_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in REMOVABLE_SUFFIXES or relative in REMOVABLE_ZIPS:
            selected.append(path)
    return sorted(selected, key=lambda item: item.as_posix())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    results_root = (root / "results").resolve()
    manifest_path = (root / args.manifest).resolve()
    receipt_path = (root / args.receipt).resolve()
    if root not in manifest_path.parents or root not in receipt_path.parents:
        raise SystemExit("manifest and receipt must remain inside the repository")
    if manifest_path.exists() or receipt_path.exists():
        raise SystemExit("refusing to overwrite an existing cleanup artifact")

    tracked = tracked_paths(root)
    selected = targets(root)
    records: list[dict[str, object]] = []
    for index, path in enumerate(selected, start=1):
        resolved = path.resolve()
        if results_root not in resolved.parents:
            raise SystemExit(f"target escaped results root: {resolved}")
        relative = resolved.relative_to(root).as_posix()
        if relative in tracked:
            raise SystemExit(f"refusing tracked target: {relative}")
        stat = resolved.stat()
        records.append(
            {
                "mtime_ns_before": stat.st_mtime_ns,
                "path": relative,
                "sha256": sha256_file(resolved),
                "size_bytes": stat.st_size,
            }
        )
        if index % 25 == 0 or index == len(selected):
            print(f"hashed {index}/{len(selected)}", flush=True)

    manifest = {
        "format": "abi-results-storage-cleanup-manifest/1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "removable_suffixes": sorted(REMOVABLE_SUFFIXES),
            "removable_zips": sorted(REMOVABLE_ZIPS),
            "tracked_targets_permitted": 0,
        },
        "results_root": "results",
        "target_count": len(records),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in records),
        "targets": records,
    }
    atomic_json(manifest_path, manifest)
    print(f"manifest={manifest_path}", flush=True)

    if not args.execute:
        return 0
    if args.confirm != CONFIRMATION:
        raise SystemExit(f"--execute requires --confirm {CONFIRMATION}")

    removed_bytes = 0
    for index, row in enumerate(records, start=1):
        path = (root / str(row["path"])).resolve()
        if results_root not in path.parents:
            raise SystemExit(f"target escaped results root during removal: {path}")
        stat = path.stat()
        if stat.st_size != row["size_bytes"] or stat.st_mtime_ns != row["mtime_ns_before"]:
            raise SystemExit(f"target changed after hashing; refusing removal: {path}")
        path.unlink()
        removed_bytes += stat.st_size
        if index % 25 == 0 or index == len(records):
            print(f"removed {index}/{len(records)}", flush=True)

    receipt = {
        "format": "abi-results-storage-cleanup-receipt/1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "removed_count": len(records),
        "removed_size_bytes": removed_bytes,
        "postcondition_all_targets_absent": all(
            not (root / str(row["path"])).exists() for row in records
        ),
    }
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
