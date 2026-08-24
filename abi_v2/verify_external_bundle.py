"""Verify an extracted ABI V2 clean-room bundle without model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


class ExternalBundleError(RuntimeError):
    """Raised when a clean-room bundle fails content verification."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(root: Path, *, strict: bool = False) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ExternalBundleError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "abi-v2-clean-room-manifest/1":
        raise ExternalBundleError("unsupported clean-room manifest")

    expected: set[str] = {"manifest.json"}
    failures: list[dict[str, Any]] = []
    for record in manifest.get("files", []):
        relative = str(record["path"])
        expected.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ExternalBundleError(f"manifest path escapes bundle: {relative}") from exc
        if not path.is_file():
            failures.append({"path": relative, "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_sha256 = _sha256(path)
        if actual_size != int(record["bytes"]) or actual_sha256 != record["sha256"]:
            failures.append(
                {
                    "path": relative,
                    "reason": "identity_mismatch",
                    "expected_bytes": int(record["bytes"]),
                    "actual_bytes": actual_size,
                    "expected_sha256": record["sha256"],
                    "actual_sha256": actual_sha256,
                }
            )

    unexpected: list[str] = []
    if strict:
        actual = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        }
        unexpected = sorted(actual - expected)
    passed = not failures and not unexpected
    return {
        "format": "abi-v2-external-bundle-verification/1",
        "status": "PASS_EXACT_BUNDLE_IDENTITY" if passed else "FAIL_BUNDLE_IDENTITY",
        "passed": passed,
        "manifest_sha256": _sha256(manifest_path),
        "files_expected": len(expected) - 1,
        "files_verified": len(expected) - 1 - len(failures),
        "failures": failures,
        "unexpected_files": unexpected,
        "claim_boundary": "This verifies archive identity only; it is not matrix execution or independent reproduction evidence.",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = verify_bundle(Path(args.root), strict=args.strict)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
