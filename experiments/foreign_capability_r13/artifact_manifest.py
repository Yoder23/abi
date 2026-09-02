"""Build a content-addressed inventory of local R13 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.native_isa_r11.core import sha256_file

from .core import R13Error, evidence_hash, write_json_once


def _publication_class(relative: str) -> str:
    if relative.endswith(".safetensors") or relative.endswith("observations.jsonl"):
        return "REQUIRED_EXTERNAL_RELEASE_ASSET"
    if "/logs/" in f"/{relative}" or "/worker_logs/" in f"/{relative}":
        return "OPTIONAL_DIAGNOSTIC"
    return "COMPACT_GIT_EVIDENCE"


def build(results_root: Path, output: Path) -> dict:
    if output.exists():
        raise R13Error(f"immutable artifact manifest exists: {output}")
    files = []
    for path in sorted(results_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path == output or "private" in path.relative_to(results_root).parts:
            continue
        relative = path.relative_to(results_root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "publication_class": _publication_class(relative),
            }
        )
    result = {
        "format": "abi-r13-local-artifact-manifest/1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "required_external_release_bytes": sum(
            item["bytes"]
            for item in files
            if item["publication_class"] == "REQUIRED_EXTERNAL_RELEASE_ASSET"
        ),
        "public_release_published": False,
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    try:
        result = build(Path(args.results_root).resolve(), output)
        write_json_once(output, result)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
