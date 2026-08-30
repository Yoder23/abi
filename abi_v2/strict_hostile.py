"""Hostile fail-closed tests for a marked disposable repaired release tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

from .canonical import canonical_json_bytes, sha256_bytes
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


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    value["evidence_sha256"] = evidence_hash(value)
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    missing_file(
        "missing_required_catalog",
        Path("catalogs/capability_compiler_phase1_frozen_v1.json"),
    )

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
        "results/abi_final_validation_v2/live_causality_r5_source_bound/qwen2/observations.jsonl"
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

    manifest_target = root / "results/abi_final_validation_v2/live_causality_r5_source_bound/qwen2/manifest.json"
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

    isolated_source = root / "abi_v2/isolated_certification.py"
    isolated_source_backup = backup_root / "isolated-source.backup"
    shutil.copy2(isolated_source, isolated_source_backup)
    rows.append(
        _expect_rejected(
            root,
            name="stale_isolated_execution_source",
            mutate=lambda: isolated_source.write_bytes(isolated_source.read_bytes() + b"\n"),
            restore=lambda: shutil.copy2(isolated_source_backup, isolated_source),
        )
    )

    transitive_source = (
        root.parent / "layercake_release/layercake/routing/catalog_router.py"
    )
    transitive_source_backup = backup_root / "transitive-source.backup"
    shutil.copy2(transitive_source, transitive_source_backup)
    rows.append(
        _expect_rejected(
            root,
            name="stale_transitive_execution_source",
            mutate=lambda: transitive_source.write_bytes(
                transitive_source.read_bytes() + b"\n"
            ),
            restore=lambda: shutil.copy2(transitive_source_backup, transitive_source),
        )
    )

    missing_file(
        "missing_raw_mount_table",
        Path(
            "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
            "pythia/mountinfo.txt"
        ),
    )
    missing_file(
        "missing_reachable_filesystem_inventory",
        Path(
            "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
            "pythia/reachable-filesystem-inventory.jsonl"
        ),
    )
    missing_file(
        "missing_frozen_adapter",
        Path(
            "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
            "layercake/certification/adapter.json"
        ),
    )

    receipt_target = root / (
        "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
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

    inventory_base = root / (
        "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/layercake"
    )
    inventory_target = inventory_base / "reachable-filesystem-inventory.jsonl"
    inventory_isolation = inventory_base / "physical-isolation.json"
    inventory_result = inventory_base / "certification/result.json"
    inventory_receipt = inventory_base / "receipt.json"
    inventory_backups = {
        inventory_target: backup_root / "filesystem-inventory.backup",
        inventory_isolation: backup_root / "filesystem-isolation.backup",
        inventory_result: backup_root / "filesystem-result.backup",
        inventory_receipt: backup_root / "filesystem-receipt.backup",
    }
    for target, backup in inventory_backups.items():
        shutil.copy2(target, backup)

    def rehash_removed_filesystem_row() -> None:
        inventory_rows = [
            json.loads(line)
            for line in inventory_target.read_text(encoding="utf-8").splitlines()
        ]
        inventory_rows = [row for row in inventory_rows if row.get("path") != "/bin"]
        inventory_target.write_bytes(
            b"".join(canonical_json_bytes(row) for row in inventory_rows)
        )
        isolation = _json(inventory_isolation)
        summary = isolation["reachable_filesystem_forbidden_scan"]
        summary["inventory_rows"] = int(summary["inventory_rows"]) - 1
        summary["symlinks_scanned"] = int(summary["symlinks_scanned"]) - 1
        summary["inventory_jsonl_sha256"] = _sha256_file(inventory_target)
        summary["evidence_sha256"] = evidence_hash(summary)
        _replace_json(inventory_isolation, _rehash(isolation))
        result = _json(inventory_result)
        result["physical_isolation"]["evidence_sha256"] = isolation["evidence_sha256"]
        _replace_json(inventory_result, _rehash(result))
        receipt = _json(inventory_receipt)
        receipt["reachable_filesystem_inventory_sha256"] = _sha256_file(
            inventory_target
        )
        receipt["isolation_evidence_sha256"] = _sha256_file(inventory_isolation)
        receipt["result_sha256"] = _sha256_file(inventory_result)
        _replace_json(inventory_receipt, _rehash(receipt))

    def restore_filesystem_row() -> None:
        for target, backup in inventory_backups.items():
            shutil.copy2(backup, target)

    rows.append(
        _expect_rejected(
            root,
            name="rehashed_missing_reachable_filesystem_row",
            mutate=rehash_removed_filesystem_row,
            restore=restore_filesystem_row,
        )
    )


    cert_result = root / (
        "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
        "qwen2/certification/result.json"
    )
    cert_receipt = root / (
        "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
        "qwen2/receipt.json"
    )
    cert_result_backup = backup_root / "cert-result.backup"
    cert_receipt_backup = backup_root / "cert-receipt.backup"
    shutil.copy2(cert_result, cert_result_backup)
    shutil.copy2(cert_receipt, cert_receipt_backup)

    def rehash_fabricated_certification_row() -> None:
        value = _json(cert_result)
        value["checks"]["roundtrip_rows"][0]["input_utf8_sha256"] = "0" * 64
        _replace_json(cert_result, _rehash(value))
        receipt = _json(cert_receipt)
        receipt["result_sha256"] = _sha256_file(cert_result)
        _replace_json(cert_receipt, _rehash(receipt))

    def restore_certification_row() -> None:
        shutil.copy2(cert_result_backup, cert_result)
        shutil.copy2(cert_receipt_backup, cert_receipt)

    def rehash_removed_native_forward_rows() -> None:
        value = _json(cert_result)
        checks = value["checks"]
        checks["native_forward_rows"] = []
        checks["native_forward_records"] = 0
        checks["native_forward_finite_records"] = 0
        checks["native_argmax_id_hashes"] = []
        _replace_json(cert_result, _rehash(value))
        receipt = _json(cert_receipt)
        receipt["result_sha256"] = _sha256_file(cert_result)
        _replace_json(cert_receipt, _rehash(receipt))

    rows.append(
        _expect_rejected(
            root,
            name="rehashed_missing_native_forward_rows",
            mutate=rehash_removed_native_forward_rows,
            restore=restore_certification_row,
        )
    )

    rows.append(
        _expect_rejected(
            root,
            name="rehashed_fabricated_certification_row",
            mutate=rehash_fabricated_certification_row,
            restore=restore_certification_row,
        )
    )

    capsule_manifest = root / (
        "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
        "layercake/certification-capsule-manifest.json"
    )
    capsule_receipt = root / (
        "results/abi_final_validation_v2/isolated_certification_strict_r5_recursive_bound/"
        "layercake/receipt.json"
    )
    capsule_manifest_backup = backup_root / "capsule-manifest.backup"
    capsule_receipt_backup = backup_root / "capsule-receipt.backup"
    shutil.copy2(capsule_manifest, capsule_manifest_backup)
    shutil.copy2(capsule_receipt, capsule_receipt_backup)

    def rehash_capsule_classification() -> None:
        value = _json(capsule_manifest)
        value["files"][0]["classification"] = "host_checkpoint"
        _replace_json(capsule_manifest, _rehash(value))
        receipt = _json(capsule_receipt)
        receipt["capsule_manifest_sha256"] = _sha256_file(capsule_manifest)
        _replace_json(capsule_receipt, _rehash(receipt))

    def restore_capsule_classification() -> None:
        shutil.copy2(capsule_manifest_backup, capsule_manifest)
        shutil.copy2(capsule_receipt_backup, capsule_receipt)

    rows.append(
        _expect_rejected(
            root,
            name="rehashed_capsule_classification",
            mutate=rehash_capsule_classification,
            restore=restore_capsule_classification,
        )
    )

    missing_file(
        "missing_condition_receipt",
        Path(
            "results/abi_final_validation_v2/live_causality_r5_source_bound/"
            "qwen2/conditions/zero_state.json"
        ),
    )

    causal_base = root / "results/abi_final_validation_v2/live_causality_r5_source_bound/qwen2"
    causal_manifest = causal_base / "manifest.json"
    causal_observations = causal_base / "observations.jsonl"
    random_receipt = causal_base / "conditions/random_state.json"
    causal_manifest_backup = backup_root / "causal-manifest.backup"
    causal_observations_backup = backup_root / "causal-observations.backup"
    random_receipt_backup = backup_root / "random-receipt.backup"
    shutil.copy2(causal_manifest, causal_manifest_backup)
    shutil.copy2(causal_observations, causal_observations_backup)
    shutil.copy2(random_receipt, random_receipt_backup)

    def rehash_random_intervention() -> None:
        receipt = _json(random_receipt)
        intervention = receipt["intervention"]
        intervention["values_after"][0] += 0.001
        intervention["after_sha256"] = sha256_bytes(
            canonical_json_bytes(intervention["values_after"])
        )
        intervention["intervention_sha256"] = sha256_bytes(
            canonical_json_bytes(
                {key: value for key, value in intervention.items() if key != "intervention_sha256"}
            )
        )
        raw_rows = [
            json.loads(line)
            for line in causal_observations.read_text(encoding="utf-8").splitlines()
        ]
        condition_rows = []
        for row in raw_rows:
            if row["condition"] != "random_state":
                continue
            row["host_state"]["intervention_sha256"] = intervention[
                "intervention_sha256"
            ]
            state_hash = sha256_bytes(canonical_json_bytes(row["host_state"]))
            row["host_state_sha256"] = state_hash
            row["applied_host_state_sha256"] = state_hash
            condition_rows.append(row)
        causal_observations.write_bytes(
            b"".join(canonical_json_bytes(row) for row in raw_rows)
        )
        receipt["observations_sha256"] = hashlib.sha256(
            b"".join(canonical_json_bytes(row) for row in condition_rows)
        ).hexdigest()
        _replace_json(random_receipt, _rehash(receipt))
        manifest = _json(causal_manifest)
        manifest["observations_sha256"] = _sha256_file(causal_observations)
        process = next(
            row
            for row in manifest["condition_processes"]
            if row["condition"] == "random_state"
        )
        process["observations_sha256"] = receipt["observations_sha256"]
        process["receipt_sha256"] = _sha256_file(random_receipt)
        _replace_json(causal_manifest, _rehash(manifest))

    def restore_random_intervention() -> None:
        shutil.copy2(causal_manifest_backup, causal_manifest)
        shutil.copy2(causal_observations_backup, causal_observations)
        shutil.copy2(random_receipt_backup, random_receipt)

    rows.append(
        _expect_rejected(
            root,
            name="rehashed_nondeterministic_random_intervention",
            mutate=rehash_random_intervention,
            restore=restore_random_intervention,
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
        "format": "abi-v2-strict-hostile-verification/3",
        "status": "PASS_STRICT_VERIFIER_FAILS_CLOSED" if passed else "FAIL_STRICT_HOSTILE",
        "baseline_evidence_sha256": baseline_sha256,
        "post_restore_evidence_sha256": repaired_sha256,
        "mutations": rows,
        "mutations_rejected": sum(int(row["rejected"]) for row in rows),
        "mutations_required": len(rows),
        "trusted_scientific_boolean_dependencies": forbidden_dependencies,
        "source_tree_restored": baseline_sha256 == repaired_sha256,
        "strict_verifier_source_sha256": _sha256_file(
            root / "abi_v2/strict_validation.py"
        ),
        "hostile_verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
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
