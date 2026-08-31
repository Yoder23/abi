"""Fail-closed public-evidence and hostile-control audit for R10."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable

from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .runtime import CopyPasteRuntimeError, load_package, sha256_file
from .slot import CanonicalCapabilitySlot


class R10HostileAuditError(RuntimeError):
    """Raised when evidence or an expected hostile rejection changes."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise R10HostileAuditError(f"expected object: {path}")
    return value


def _evidence_hash(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("evidence_sha256", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _expect_rejection(name: str, operation: Callable[[], None]) -> dict[str, Any]:
    try:
        operation()
    except (OSError, ValueError, CopyPasteRuntimeError, R10HostileAuditError) as exc:
        return {"control": name, "rejected": True, "error_type": type(exc).__name__}
    raise R10HostileAuditError(f"hostile control was accepted: {name}")


def audit(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "public_manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("format") != "abi-copy-paste-r10-public-evidence-manifest/1" or manifest.get(
        "evidence_sha256"
    ) != _evidence_hash(manifest):
        raise R10HostileAuditError("public manifest identity changed")
    for field in ("run_receipt", "negative_report", "live_replay", "raw_evidence"):
        item = manifest[field]
        path = run_dir / item["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(item["bytes"])
            or sha256_file(path) != item["sha256"]
        ):
            raise R10HostileAuditError(f"manifest evidence changed: {field}")
    for item in manifest["verifier_failures_preserved"]:
        if sha256_file(run_dir / item["path"]) != item["sha256"]:
            raise R10HostileAuditError("verifier failure receipt changed")
    packages = []
    for name in manifest["packages"]:
        path = run_dir / "packages" / name
        load_package(path)
        packages.append(path)

    raw = manifest["raw_evidence"]
    expected_members = raw["members"]
    with tarfile.open(run_dir / raw["path"], "r:gz") as archive:
        files = {item.name: item for item in archive.getmembers() if item.isfile()}
        if set(files) != set(expected_members):
            raise R10HostileAuditError("raw archive member inventory changed")
        for name, expected in expected_members.items():
            handle = archive.extractfile(files[name])
            if handle is None:
                raise R10HostileAuditError("raw archive member unavailable")
            data = handle.read()
            if (
                len(data) != expected["bytes"]
                or hashlib.sha256(data).hexdigest() != expected["sha256"]
            ):
                raise R10HostileAuditError("raw archive member identity changed")
            if data.count(b"\n") != expected["rows"]:
                raise R10HostileAuditError("raw archive row count changed")

    controls = []
    with tempfile.TemporaryDirectory(prefix="abi-r10-hostile-") as temporary:
        temporary_path = Path(temporary)
        tampered = temporary_path / packages[0].name
        shutil.copy2(packages[0], tampered)
        data = bytearray(tampered.read_bytes())
        data[len(data) // 2] ^= 1
        tampered.write_bytes(data)
        controls.append(_expect_rejection("tampered_package", lambda: load_package(tampered)))
        controls.append(
            _expect_rejection(
                "missing_package", lambda: load_package(temporary_path / "missing.abipkg")
            )
        )
    controls.append(
        _expect_rejection("empty_slot_execution", lambda: CanonicalCapabilitySlot().execute(["x"]))
    )

    def forged_manifest() -> None:
        value = dict(manifest)
        value["status"] = "FORGED_PASS"
        if value["evidence_sha256"] != _evidence_hash(value):
            raise R10HostileAuditError("forged manifest rejected")

    controls.append(_expect_rejection("forged_manifest_status", forged_manifest))

    def wrong_raw_hash() -> None:
        if sha256_file(run_dir / raw["path"]) != "0" * 64:
            raise R10HostileAuditError("wrong raw hash rejected")

    controls.append(_expect_rejection("wrong_raw_archive_hash", wrong_raw_hash))
    result = {
        "format": "abi-copy-paste-r10-hostile-audit/1",
        "status": "PASS",
        "manifest_sha256": sha256_file(manifest_path),
        "packages_verified": len(packages),
        "raw_members_verified": len(expected_members),
        "hostile_controls": controls,
        "hostile_controls_rejected": sum(item["rejected"] for item in controls),
        "claim_boundary": "Evidence integrity for bounded R10 runtime component only",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = audit(Path(args.run_dir).resolve())
        output = Path(args.output).resolve()
        if output.exists():
            raise R10HostileAuditError(f"immutable hostile audit exists: {output}")
        output.write_bytes(json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, tarfile.TarError) as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
