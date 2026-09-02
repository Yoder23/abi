"""Create a content-addressed inventory of local R14 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import evidence_hash, sha256_file, write_json_once


def build(root: Path, output: Path) -> dict[str, Any]:
    files = []
    for directory in (root / "heldout_v1", root / "live_replay_v1"):
        for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file():
                files.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    bulk_suffixes = (".safetensors", ".jsonl", ".txt")
    bulk = [item for item in files if str(item["path"]).endswith(bulk_suffixes)]
    result = {
        "format": "abi-r14-local-artifact-manifest/1",
        "scope": "LOCAL_NEGATIVE_EVIDENCE_NOT_PUBLIC_REPRODUCTION",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
        "bulk_files": len(bulk),
        "bulk_bytes": sum(int(item["bytes"]) for item in bulk),
        "bulk_publication_status": "NOT_PUBLISHED",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = build(Path(args.root).resolve(), Path(args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
