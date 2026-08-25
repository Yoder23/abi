"""Hostile fail-closed tests for a marked disposable repaired release tree."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from .final_validation import CAPABILITY_PATHS, evidence_hash
from .strict_validation import StrictValidationError, verify

MARKER = ".abi-disposable-validation-root"


class StrictHostileError(RuntimeError):
    """Raised when destructive hostile tests are not safely scoped."""


def _write(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise StrictHostileError(f"immutable hostile receipt exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StrictHostileError(f"expected object: {path}")
    return value


def _replace_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _require_disposable(root: Path) -> None:
    root = root.resolve()
    marker = root / MARKER
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != (
        "ABI_DISPOSABLE_HOSTILE_ROOT"
    ):
        raise StrictHostileError("hostile root lacks the exact disposable marker")
    if (root / ".git").exists():
        raise StrictHostileError("refusing to mutate a Git working tree")
    if not (root / "abi_v2/strict_validation.py").is_file():
        raise StrictHostileError("hostile root is not a repaired ABI release tree")


def _expect_rejected(
    root: Path,
    *,
    name: str,
    mutate: Callable[[], None],
    restore: Callable[[], None],
) -> dict[str, Any]:
    try:
        mutate()
        try:
            verify(root)
        except StrictValidationError as exc:
            return {
                "mutation": name,
                "rejected": True,
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        return {
            "mutation": name,
            "rejected": False,
            "exception_type": None,
            "message": "strict verifier incorrectly accepted mutation",
        }
    finally:
        restore()


def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _require_disposable(root)
    baseline = verify(root)
    baseline_sha256 = evidence_hash(baseline)
    backup_root = root / ".hostile-backups"
    if backup_root.exists():
        raise StrictHostileError("hostile backup directory already exists")
    backup_root.mkdir()
    rows = []

    def missing_file(name: str, relative: Path) -> None:
        target = root / relative
        backup = backup_root / name
        if not target.is_file():
            raise StrictHostileError(f"mutation target missing before test: {target}")
        rows.append(
            _expect_rejected(
                root,
                name=name,
                mutate=lambda: target.replace(backup),
                restore=lambda: backup.replace(target),
            )
        )

    missing_file("missing_capability_package", CAPABILITY_PATHS["python"])

    corrupt_target = root / CAPABILITY_PATHS["python"]
    corrupt_backup = backup_root / "corrupt-package.backup"
    shutil.copy2(corrupt_target, corrupt_backup)

    def corrupt_package() -> None:
        payload = bytearray(corrupt_target.read_bytes())
        payload[len(payload) // 2] ^= 1
        corrupt_target.write_bytes(payload)

    rows.append(
        _expect_rejected(
            root,
            name="corrupt_capability_package",
            mutate=corrupt_package,
            restore=lambda: shutil.copy2(corrupt_backup, corrupt_target),
        )
    )

    causal_rows = Path(
        "results/abi_final_validation_v2/live_causality/qwen2/observations.jsonl"
    )
    missing_file("missing_raw_causality_file", causal_rows)
    raw_target = root / causal_rows
    raw_backup = backup_root / "raw-row.backup"
    shutil.copy2(raw_target, raw_backup)

    def remove_raw_row() -> None:
        lines = raw_target.read_bytes().splitlines(keepends=True)
        raw_target.write_bytes(b"".join(lines[:-1]))

    rows.append(
        _expect_rejected(
            root,
            name="missing_raw_causality_row",
            mutate=remove_raw_row,
            restore=lambda: shutil.copy2(raw_backup, raw_target),
        )
    )

    manifest_target = root / "results/abi_final_validation_v2/live_causality/qwen2/manifest.json"
    manifest_backup = backup_root / "manifest.backup"
    shutil.copy2(manifest_target, manifest_backup)

    def remove_required_hash() -> None:
        value = _json(manifest_target)
        value.pop("observations_sha256", None)
        value["evidence_sha256"] = evidence_hash(value)
        _replace_json(manifest_target, value)

    rows.append(
        _expect_rejected(
            root,
            name="missing_required_raw_hash",
            mutate=remove_required_hash,
            restore=lambda: shutil.copy2(manifest_backup, manifest_target),
        )
    )

    source_target = root / "abi_v2/live_causality.py"
    source_backup = backup_root / "live-source.backup"
    shutil.copy2(source_target, source_backup)
    rows.append(
        _expect_rejected(
            root,
            name="stale_execution_source",
            mutate=lambda: source_target.write_bytes(source_target.read_bytes() + b"\n"),
            restore=lambda: shutil.copy2(source_backup, source_target),
        )
    )

    missing_file(
        "missing_raw_mount_table",
        Path(
            "results/abi_final_validation_v2/isolated_certification_strict/"
            "pythia/mountinfo.txt"
        ),
    )
    missing_file(
        "missing_frozen_adapter",
        Path(
            "results/abi_final_validation_v2/isolated_certification_strict/"
            "layercake/certification/adapter.json"
        ),
    )

    receipt_target = root / (
        "results/abi_final_validation_v2/isolated_certification_strict/"
        "qwen2/receipt.json"
    )
    receipt_backup = backup_root / "receipt.backup"
    shutil.copy2(receipt_target, receipt_backup)

    def stale_receipt_hash() -> None:
        value = _json(receipt_target)
        value["result_sha256"] = "0" * 64
        value["evidence_sha256"] = evidence_hash(value)
        _replace_json(receipt_target, value)

    rows.append(
        _expect_rejected(
            root,
            name="stale_certification_binding",
            mutate=stale_receipt_hash,
            restore=lambda: shutil.copy2(receipt_backup, receipt_target),
        )
    )

    repaired = verify(root)
    repaired_sha256 = evidence_hash(repaired)
    verifier_source = (root / "abi_v2/strict_validation.py").read_text(encoding="utf-8")
    forbidden_dependencies = [
        token
        for token in (
            'result["gates"]',
            "result['gates']",
            'result.get("status")',
            "result.get('status')",
            'manifest.get("status")',
            "manifest.get('status')",
        )
        if token in verifier_source
    ]
    passed = all(row["rejected"] for row in rows) and not forbidden_dependencies
    result = {
        "format": "abi-v2-strict-hostile-verification/1",
        "status": "PASS_STRICT_VERIFIER_FAILS_CLOSED" if passed else "FAIL_STRICT_HOSTILE",
        "baseline_evidence_sha256": baseline_sha256,
        "post_restore_evidence_sha256": repaired_sha256,
        "mutations": rows,
        "mutations_rejected": sum(int(row["rejected"]) for row in rows),
        "mutations_required": len(rows),
        "trusted_scientific_boolean_dependencies": forbidden_dependencies,
        "source_tree_restored": baseline_sha256 == repaired_sha256,
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    value = run(Path(args.root))
    if args.output:
        _write(Path(args.output).resolve(), value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
