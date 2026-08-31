"""Create an immutable content manifest for the complete R11 evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .core import R11Error, sha256_file


def build(evidence_root: Path, output: Path) -> dict:
    if output.exists():
        raise R11Error(f"immutable R11 manifest exists: {output}")
    root = evidence_root.resolve()
    files = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.resolve() == output.resolve():
            continue
        relative = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "format": "abi-native-neural-isa-r11-evidence-manifest/1",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = build(Path(args.evidence_root), Path(args.output))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
