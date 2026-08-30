"""Fail-closed recomputation for the repaired ABI final validation.

No scientific status/gate boolean produced by an experiment is accepted as
evidence.  This verifier derives claims from immutable files, raw observation
rows, hashes, counts, outputs, timings, and live failure records.  Missing,
extra, stale, or unrecomputable inputs are fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from abi.capability_compiler_phase2_common import evaluate_functional

from .canonical import canonical_json_bytes, sha256_bytes, verify_reference
from .capability_matrix import (
    CAPABILITIES,
    DOMAINS,
    _matrix_records,
    _source_references,
)
from .execution_sources import ExecutionSourceError, execution_source_manifest
from .final_validation import CAPABILITY_PATHS, HOSTS, MATRIX_DIRS, evidence_hash
from .host_certification import _neutral_texts
from .isolated_certification import (
    ALLOWED_CLASSIFICATIONS,
    FILESYSTEM_INVENTORY_FORMAT,
    _mount_rows,
)
from .live_causality import CONDITIONS, SAMPLE_SEED, _selected

EVIDENCE_ROOT = Path("results/abi_final_validation_v2")
CERTIFICATION_ROOT = EVIDENCE_ROOT / "isolated_certification_strict_r6_full_stream_bound"
CAUSALITY_ROOT = EVIDENCE_ROOT / "live_causality_r6_source_bound"
ISOLATION_ROOT = EVIDENCE_ROOT / "live_isolation_r6_source_bound"
FORBIDDEN_CAPABILITY_SUFFIXES = {".abi", ".cake", ".abix", ".abicir"}

# These are release-source commitments, not experiment-supplied status flags.
# They make the complete raw reachable-filesystem inventory immutable: deleting
# or fabricating even an otherwise ordinary system-runtime row cannot be hidden
# by recomputing the enclosing JSON hashes.
EXPECTED_REACHABLE_INVENTORIES = {
    "layercake": {
        "rows": 100_511,
        "sha256": "cf3ddd4c4a91bdec7fa3a3b40718182a0b92c0a61ddc9f9955bab8df5ff120a5",
    },
    "qwen2": {
        "rows": 100_517,
        "sha256": "4b3fe52f50e660de361d4717b77d40fd016635b68700001084b09ff7bea795d6",
    },
    "pythia": {
        "rows": 100_515,
        "sha256": "c26abad912047399b7065865dd89dcb50bf3375a161ce96032f6e1544adf7e7b",
    },
}


class StrictValidationError(RuntimeError):
    """Raised whenever a required claim cannot be independently recomputed."""


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise StrictValidationError(f"required file missing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise StrictValidationError(f"required file unreadable: {path}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictValidationError(f"required JSON unavailable: {path}") from exc
    if not isinstance(value, dict):
        raise StrictValidationError(f"required JSON object changed: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise StrictValidationError(f"required JSONL unavailable: {path}") from exc
    if not lines:
        raise StrictValidationError(f"required JSONL is empty: {path}")
    rows = []
    for position, line in enumerate(lines):
        if not line.strip():
            raise StrictValidationError(f"blank raw row at {path}:{position + 1}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StrictValidationError(
                f"invalid raw row at {path}:{position + 1}"
            ) from exc
        if not isinstance(row, dict):
            raise StrictValidationError(f"non-object raw row at {path}:{position + 1}")
        rows.append(row)
    return rows


def read_text(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise StrictValidationError(f"required text unavailable: {path}") from exc
    if not value:
        raise StrictValidationError(f"required text is empty: {path}")
    return value


def verify_evidence_hash(value: Mapping[str, Any], *, label: str) -> None:
    if value.get("evidence_sha256") != evidence_hash(value):
        raise StrictValidationError(f"stale or invalid evidence hash: {label}")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _inventory_classification(path: str, runtime_site: str) -> str | None:
    if path == "/capsule" or path.startswith("/capsule/"):
        return "certification_capsule"
    if path == "/usr" or path.startswith("/usr/"):
        return "python_system_runtime"
    if path == "/etc" or path.startswith("/etc/"):
        return "system_runtime_configuration"
    if path == runtime_site or path.startswith(runtime_site.rstrip("/") + "/"):
        return "python_package_runtime"
    if path in {"/bin", "/sbin", "/lib", "/lib64"}:
        return "runtime_alias"
    return None


def verify_reachable_filesystem_inventory(
    *,
    host: str,
    path: Path,
    summary: Mapping[str, Any],
    runtime_site: str,
) -> dict[str, dict[str, Any]]:
    """Recompute the content-bound reachable-root inventory from raw rows."""

    if summary.get("format") != FILESYSTEM_INVENTORY_FORMAT:
        raise StrictValidationError(f"reachable inventory format changed: {host}")
    verify_evidence_hash(summary, label=f"{host}/reachable-filesystem-inventory")
    rows = read_jsonl(path)
    expected_bytes = b"".join(canonical_json_bytes(row) for row in rows)
    if path.read_bytes() != expected_bytes:
        raise StrictValidationError(f"reachable inventory JSONL is noncanonical: {host}")
    commitment = EXPECTED_REACHABLE_INVENTORIES.get(host)
    if (
        commitment is None
        or len(rows) != commitment["rows"]
        or hashlib.sha256(expected_bytes).hexdigest() != commitment["sha256"]
    ):
        raise StrictValidationError(f"reachable inventory release commitment changed: {host}")
    if summary.get("inventory_jsonl_sha256") != hashlib.sha256(expected_bytes).hexdigest():
        raise StrictValidationError(f"reachable inventory hash changed: {host}")
    paths = [row.get("path") for row in rows]
    if (
        any(not isinstance(value, str) or not value.startswith("/") for value in paths)
        or paths != sorted(paths)
        or len(paths) != len(set(paths))
    ):
        raise StrictValidationError(f"reachable inventory paths changed: {host}")

    regular_schema = {
        "path",
        "kind",
        "mount_classification",
        "bytes",
        "sha256",
        "content_bytes_scanned",
        "campaign_identifier_matches",
        "embedded_archive_members_scanned",
        "embedded_archive_uncompressed_bytes_scanned",
        "embedded_campaign_identifier_matches",
        "embedded_archive_maximum_depth",
        "unsupported_archive_signatures",
        "capability_archive_signatures",
        "forbidden_archive_member_paths",
    }
    symlink_schema = {
        "path",
        "kind",
        "mount_classification",
        "link_target",
        "resolved_path",
        "target_state",
        "sha256",
    }
    regular_rows = []
    symlink_rows = []
    allowed_link_roots = ("/usr", "/etc", runtime_site, "/dev", "/proc", "/capsule")
    for row in rows:
        row_path = str(row["path"])
        expected_classification = _inventory_classification(row_path, runtime_site)
        if (
            expected_classification is None
            or row.get("mount_classification") != expected_classification
            or not _valid_sha256(row.get("sha256"))
        ):
            raise StrictValidationError(f"unclassified reachable inventory row: {host}/{row_path}")
        if row.get("kind") == "regular_file":
            if set(row) != regular_schema:
                raise StrictValidationError(f"regular inventory schema changed: {host}/{row_path}")
            numeric = (
                "bytes",
                "content_bytes_scanned",
                "campaign_identifier_matches",
                "embedded_archive_members_scanned",
                "embedded_archive_uncompressed_bytes_scanned",
                "embedded_campaign_identifier_matches",
                "embedded_archive_maximum_depth",
            )
            try:
                values = {field: int(row[field]) for field in numeric}
            except (KeyError, TypeError, ValueError) as exc:
                raise StrictValidationError(
                    f"reachable inventory numeric field changed: {host}/{row_path}"
                ) from exc
            if (
                any(value < 0 for value in values.values())
                or values["content_bytes_scanned"] != values["bytes"]
                or values["campaign_identifier_matches"] != 0
                or values["embedded_campaign_identifier_matches"] != 0
                or row.get("unsupported_archive_signatures") != []
                or row.get("capability_archive_signatures") != []
                or row.get("forbidden_archive_member_paths") != []
            ):
                raise StrictValidationError(f"forbidden reachable content detected: {host}/{row_path}")
            regular_rows.append(row)
        elif row.get("kind") == "symlink":
            if set(row) != symlink_schema:
                raise StrictValidationError(f"symlink inventory schema changed: {host}/{row_path}")
            target = row.get("link_target")
            resolved = row.get("resolved_path")
            target_state = row.get("target_state")
            target_in_admitted_root = isinstance(resolved, str) and any(
                resolved == root or resolved.startswith(root.rstrip("/") + "/")
                for root in allowed_link_roots
            )
            if (
                not isinstance(target, str)
                or not isinstance(resolved, str)
                or not resolved.startswith("/")
                or row["sha256"] != hashlib.sha256(target.encode("utf-8")).hexdigest()
                or target_state
                not in {"reachable_admitted_root", "absent_in_namespace"}
                or (target_state == "reachable_admitted_root") != target_in_admitted_root
            ):
                raise StrictValidationError(f"reachable symlink escaped policy: {host}/{row_path}")
            symlink_rows.append(row)
        else:
            raise StrictValidationError(f"reachable inventory kind changed: {host}/{row_path}")

    recomputed = {
        "inventory_rows": len(rows),
        "regular_files_scanned": len(regular_rows),
        "symlinks_scanned": len(symlink_rows),
        "regular_file_bytes": sum(int(row["bytes"]) for row in regular_rows),
        "content_bytes_scanned": sum(int(row["content_bytes_scanned"]) for row in regular_rows),
        "embedded_archive_members_scanned": sum(
            int(row["embedded_archive_members_scanned"]) for row in regular_rows
        ),
        "embedded_archive_uncompressed_bytes_scanned": sum(
            int(row["embedded_archive_uncompressed_bytes_scanned"])
            for row in regular_rows
        ),
        "embedded_archive_maximum_depth": max(
            (int(row["embedded_archive_maximum_depth"]) for row in regular_rows),
            default=0,
        ),
        "unsupported_archive_signatures": sum(
            len(row["unsupported_archive_signatures"]) for row in regular_rows
        ),
        "capability_archive_signature_matches": sum(
            len(row["capability_archive_signatures"]) for row in regular_rows
        ),
        "forbidden_archive_member_paths": sum(
            len(row["forbidden_archive_member_paths"]) for row in regular_rows
        ),
        "campaign_identifier_matches": sum(
            int(row["campaign_identifier_matches"])
            + int(row["embedded_campaign_identifier_matches"])
            for row in regular_rows
        ),
    }
    if any(summary.get(field) != value for field, value in recomputed.items()):
        raise StrictValidationError(f"reachable inventory aggregate changed: {host}")
    if (
        summary.get("scan_root") != "/"
        or summary.get("excluded_virtual_roots") != ["/dev", "/proc"]
        or summary.get("runtime_site_root") != runtime_site
        or int(summary.get("directories_scanned", 0)) <= 0
        or summary.get("forbidden_path_matches") != 0
        or summary.get("special_entries") != 0
        or len(regular_rows) <= 0
        or not {"certification_capsule", "python_system_runtime", "python_package_runtime"}
        <= {str(row["mount_classification"]) for row in regular_rows}
        or not {"/bin", "/sbin", "/lib"} <= set(paths)
    ):
        raise StrictValidationError(f"reachable filesystem coverage changed: {host}")
    return {str(row["path"]): row for row in rows}


def verify_certifications(
    root: Path, certification_root: Path | None = None
) -> dict[str, Any]:
    """Recompute certification solely from capsule and raw measurement bytes."""

    root = root.resolve()
    certification_root = (
        (root / CERTIFICATION_ROOT).resolve()
        if certification_root is None
        else certification_root.resolve()
    )
    adapter_manifest = read_json(root / "results/abi_v2/adapters/manifest.json")
    model_manifest = read_json(root / "external_reproduction/model_download_manifest.json")
    suite = read_json(root / "abi_v2/conformance_suite.json")
    spec = read_json(root / "abi_v2/canonical_spec.json")
    reference_path = root / suite["reference_vectors"]["path"]
    if sha256_file(reference_path) != suite["reference_vectors"]["sha256"]:
        raise StrictValidationError("generic reference-vector binding changed")
    reference = read_json(reference_path)
    records = reference.get("records")
    if not isinstance(records, list) or len(records) != suite["reference_vectors"]["records"]:
        raise StrictValidationError("generic reference-vector depth changed")
    for record in records:
        verify_reference(record)

    hosts: dict[str, Any] = {}
    for host in HOSTS:
        base = certification_root / host
        launcher = read_json(base / "launcher-receipt.json")
        receipt = read_json(base / "receipt.json")
        isolation = read_json(base / "physical-isolation.json")
        capsule = read_json(base / "certification-capsule-manifest.json")
        result = read_json(base / "certification/result.json")
        performance = read_json(base / "certification/performance.json")
        adapter_path = base / "certification/adapter.json"
        mount_path = base / "mountinfo.txt"
        filesystem_inventory_path = base / "reachable-filesystem-inventory.jsonl"
        mountinfo = read_text(mount_path)
        for label, value in (
            ("launcher", launcher),
            ("receipt", receipt),
            ("isolation", isolation),
            ("capsule", capsule),
            ("certification", result),
        ):
            verify_evidence_hash(value, label=f"{host}/{label}")
        bindings = {
            "result_sha256": base / "certification/result.json",
            "adapter_sha256": adapter_path,
            "performance_sha256": base / "certification/performance.json",
            "isolation_evidence_sha256": base / "physical-isolation.json",
            "capsule_manifest_sha256": base / "certification-capsule-manifest.json",
            "mountinfo_sha256": mount_path,
            "reachable_filesystem_inventory_sha256": filesystem_inventory_path,
        }
        for field, path in bindings.items():
            if receipt.get(field) != sha256_file(path):
                raise StrictValidationError(f"certification binding changed: {host}/{field}")
        if launcher.get("launcher", {}).get("exit_code") != 0:
            raise StrictValidationError(f"isolated worker exit changed: {host}")

        capsule_files = capsule.get("files")
        if not isinstance(capsule_files, list) or not capsule_files:
            raise StrictValidationError(f"capsule inventory missing: {host}")
        capsule_by_path = {str(row.get("path")): row for row in capsule_files}
        if len(capsule_by_path) != len(capsule_files):
            raise StrictValidationError(f"duplicate capsule inventory path: {host}")
        if capsule.get("allowed_classifications") != sorted(ALLOWED_CLASSIFICATIONS):
            raise StrictValidationError(f"capsule classification policy changed: {host}")
        expected_capsule_sources: dict[str, tuple[Path | None, str]] = {
            "abi_release/abi_v2/__init__.py": (root / "abi_v2/__init__.py", "adapter_certification_code"),
            "abi_release/abi_v2/canonical.py": (root / "abi_v2/canonical.py", "adapter_certification_code"),
            "abi_release/abi_v2/certification_pivot_runner.sh": (
                root / "abi_v2/certification_pivot_runner.sh",
                "adapter_certification_code",
            ),
            "abi_release/abi_v2/host_certification.py": (
                root / "abi_v2/host_certification.py",
                "adapter_certification_code",
            ),
            "abi_release/abi_v2/isolated_certification.py": (
                root / "abi_v2/isolated_certification.py",
                "adapter_certification_code",
            ),
            "abi_release/abi_v2/canonical_spec.json": (
                root / "abi_v2/canonical_spec.json",
                "abi_specification",
            ),
            "abi_release/abi_v2/conformance_suite.json": (
                root / "abi_v2/conformance_suite.json",
                "abi_specification",
            ),
            "abi_release/abi_v2/reference_vectors/core_vectors.json": (
                root / "abi_v2/reference_vectors/core_vectors.json",
                "generic_certification_corpus",
            ),
        }
        if host == "layercake":
            expected_capsule_sources[
                "layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py"
            ] = (
                root.parent
                / "layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py",
                "host_code",
            )
        else:
            model_files = model_manifest["models"][host]["files"]
            for name in model_files:
                expected_capsule_sources[f"abi_release/host_snapshot/{name}"] = (
                    None,
                    "host_checkpoint",
                )
        if set(capsule_by_path) != set(expected_capsule_sources):
            raise StrictValidationError(f"capsule allowlist changed: {host}")
        capability_count = 0
        success_ledger_count = 0
        for path, row in capsule_by_path.items():
            source, classification = expected_capsule_sources[path]
            if row.get("classification") != classification:
                raise StrictValidationError(f"capsule classification changed: {host}/{path}")
            if Path(path).suffix.casefold() in FORBIDDEN_CAPABILITY_SUFFIXES:
                capability_count += 1
            if "source_success" in path.casefold():
                success_ledger_count += 1
            if source is not None and (
                int(row.get("bytes", -1)) != source.stat().st_size
                or row.get("sha256") != sha256_file(source)
            ):
                raise StrictValidationError(f"stale executed capsule source: {host}/{path}")
            if source is None:
                name = Path(path).name
                expected_model_file = model_manifest["models"][host]["files"][name]
                if (
                    int(row.get("bytes", -1)) != int(expected_model_file["bytes"])
                    or row.get("sha256") != expected_model_file["sha256"]
                ):
                    raise StrictValidationError(f"host checkpoint capsule changed: {host}/{name}")
        if capability_count or success_ledger_count:
            raise StrictValidationError(f"forbidden payload entered certification: {host}")
        physical_inventory = isolation.get("capsule", {}).get("inventory")
        if not isinstance(physical_inventory, list):
            raise StrictValidationError(f"physical capsule inventory missing: {host}")
        physical_by_path = {str(row.get("path")): row for row in physical_inventory}
        expected_physical = {
            path: {
                "path": path,
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for path, row in capsule_by_path.items()
        }
        if physical_by_path != expected_physical:
            raise StrictValidationError(f"physical capsule bytes differ from manifest: {host}")
        mount_rows = _mount_rows(mountinfo)
        mount_by_point = {str(row["mount_point"]): row for row in mount_rows}
        runtime_site = isolation.get("mount", {}).get("runtime_site_mount")
        allowed_mounts = {
            "/",
            "/capsule",
            "/dev",
            "/dev/null",
            "/dev/random",
            "/dev/urandom",
            "/dev/zero",
            "/etc",
            "/proc",
            "/usr",
            runtime_site,
        }
        if (
            not isinstance(runtime_site, str)
            or not runtime_site.startswith("/home/")
            or not runtime_site.endswith("/site-packages")
            or set(mount_by_point) != allowed_mounts
            or isolation.get("mount", {}).get("allowed_mount_points")
            != sorted(allowed_mounts)
            or isolation.get("mount", {}).get("mounts") != mount_rows
            or mount_by_point["/"]["filesystem_type"] != "tmpfs"
            or mount_by_point["/dev"]["filesystem_type"] != "tmpfs"
            or mount_by_point["/etc"]["filesystem_type"] != "tmpfs"
            or mount_by_point["/proc"]["filesystem_type"] != "proc"
            or any(
                "ro" not in mount_by_point[path]["mount_options"]
                for path in ("/usr", "/etc", runtime_site, "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom")
            )
            or any(path.startswith("/mnt") or path.startswith("/oldroot") for path in mount_by_point)
        ):
            raise StrictValidationError(f"pivot-root filesystem policy changed: {host}")
        if isolation.get("mount", {}).get("mountinfo_sha256") != sha256_file(mount_path):
            raise StrictValidationError(f"raw mount table binding changed: {host}")
        forbidden_scan = isolation.get("reachable_filesystem_forbidden_scan")
        if not isinstance(forbidden_scan, dict):
            raise StrictValidationError(f"reachable-root forbidden scan changed: {host}")
        filesystem_by_path = verify_reachable_filesystem_inventory(
            host=host,
            path=filesystem_inventory_path,
            summary=forbidden_scan,
            runtime_site=runtime_site,
        )
        expected_capsule_inventory = {
            f"/capsule/{path}": {
                "bytes": int(row["bytes"]),
                "sha256": str(row["sha256"]),
            }
            for path, row in capsule_by_path.items()
        }
        expected_capsule_inventory[
            "/capsule/abi_release/certification-capsule-manifest.json"
        ] = {
            "bytes": (base / "certification-capsule-manifest.json").stat().st_size,
            "sha256": sha256_file(base / "certification-capsule-manifest.json"),
        }
        for inventory_path, expected_row in expected_capsule_inventory.items():
            actual_row = filesystem_by_path.get(inventory_path)
            if (
                actual_row is None
                or actual_row.get("kind") != "regular_file"
                or int(actual_row.get("bytes", -1)) != expected_row["bytes"]
                or actual_row.get("sha256") != expected_row["sha256"]
            ):
                raise StrictValidationError(
                    f"reachable inventory/capsule binding changed: {host}/{inventory_path}"
                )
        if int(isolation.get("process_id", -1)) != 1 or int(
            isolation.get("parent_process_id", -1)
        ) != 0:
            raise StrictValidationError(f"private PID namespace changed: {host}")
        executed_bindings = {
            "executed_isolated_certification_sha256": root
            / "abi_v2/isolated_certification.py",
            "executed_pivot_runner_sha256": root
            / "abi_v2/certification_pivot_runner.sh",
            "executed_host_certification_sha256": root / "abi_v2/host_certification.py",
        }
        for field, source in executed_bindings.items():
            if receipt.get(field) != sha256_file(source):
                raise StrictValidationError(f"stale certification execution source: {host}/{field}")
        expected_adapter = adapter_manifest["adapters"][host]["sha256"]
        if sha256_file(adapter_path) != expected_adapter:
            raise StrictValidationError(f"adapter bytes changed: {host}")
        adapter = read_json(adapter_path)
        if (
            int(adapter.get("trainable_parameters", -1)) != 0
            or int(adapter.get("optimizer_steps", -1)) != 0
        ):
            raise StrictValidationError(f"adapter structure changed: {host}")
        if result.get("canonical_spec_sha256") != sha256_file(root / "abi_v2/canonical_spec.json"):
            raise StrictValidationError(f"canonical spec binding changed: {host}")
        if result.get("conformance_suite_sha256") != sha256_file(
            root / "abi_v2/conformance_suite.json"
        ):
            raise StrictValidationError(f"conformance suite binding changed: {host}")
        if result.get("reference_implementation_sha256") != sha256_file(
            root / "abi_v2/canonical.py"
        ):
            raise StrictValidationError(f"reference implementation binding changed: {host}")
        if result.get("physical_isolation", {}).get("evidence_sha256") != isolation.get(
            "evidence_sha256"
        ):
            raise StrictValidationError(f"certification/isolation binding changed: {host}")

        checks = result.get("checks")
        if not isinstance(checks, dict):
            raise StrictValidationError(f"raw certification checks missing: {host}")
        roundtrips = checks.get("roundtrip_rows")
        if not isinstance(roundtrips, list) or len(roundtrips) != int(
            result["certification_data"]["examples"]
        ):
            raise StrictValidationError(f"roundtrip raw rows missing: {host}")
        expected_texts = _neutral_texts(
            int(suite["certification_data"]["generated_roundtrip_records"])
        )
        expected_hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in expected_texts]
        expected_bytes = [len(text.encode("utf-8")) for text in expected_texts]
        if len(roundtrips) != len(expected_texts) or any(
            row.get("input_utf8_sha256") != expected_hashes[position]
            or row.get("decoded_utf8_sha256") != expected_hashes[position]
            or int(row.get("input_utf8_bytes", -1)) != expected_bytes[position]
            or int(row.get("decoded_utf8_bytes", -2)) != expected_bytes[position]
            for position, row in enumerate(roundtrips)
        ):
            raise StrictValidationError(f"deterministic certification corpus changed: {host}")
        if result["certification_data"].get("example_sha256") != expected_hashes:
            raise StrictValidationError(f"certification corpus hashes changed: {host}")
        forwards = checks.get("native_forward_rows")
        if not isinstance(forwards, list):
            raise StrictValidationError(f"native forward raw rows missing: {host}")
        locked_forward_records = (
            0
            if host == "layercake"
            else int(suite["certification_data"]["model_forward_records"])
        )
        if (
            len(forwards) != locked_forward_records
            or int(checks.get("native_forward_records", -1)) != locked_forward_records
            or int(checks.get("native_forward_finite_records", -1))
            != locked_forward_records
        ):
            raise StrictValidationError(f"native forward depth changed: {host}")
        if any(
            int(row.get("position", -1)) != position
            or row.get("input_utf8_sha256") != expected_hashes[position]
            or int(row.get("input_units", 0)) <= 0
            or int(row.get("finite_values", -1)) != int(row.get("total_values", -2))
            for position, row in enumerate(forwards)
        ):
            raise StrictValidationError(f"non-finite native forward: {host}")
        if checks.get("native_argmax_id_hashes", []) != [
            row["argmax_id_sha256"] for row in forwards
        ]:
            raise StrictValidationError(f"native forward hashes changed: {host}")
        expected_snapshot = [] if host == "layercake" else [
            {
                "name": name,
                "bytes": int(binding["bytes"]),
                "sha256": binding["sha256"],
            }
            for name, binding in sorted(model_manifest["models"][host]["files"].items())
        ]
        if result.get("snapshot_inventory") != expected_snapshot:
            raise StrictValidationError(f"certification snapshot inventory changed: {host}")

        alone = [float(value) for value in performance.get("host_alone_seconds", [])]
        adapted = [float(value) for value in performance.get("host_plus_idle_adapter_seconds", [])]
        minimum = int(spec["performance_gate"]["minimum_repeated_observations"])
        if len(alone) != len(adapted) or len(alone) < minimum or any(
            not math.isfinite(value) or value <= 0 for value in (*alone, *adapted)
        ):
            raise StrictValidationError(f"performance raw rows incomplete: {host}")
        baseline = statistics.median(alone)
        with_adapter = statistics.median(adapted)
        overhead = with_adapter / baseline - 1.0
        threshold = float(spec["performance_gate"]["maximum_overhead_fraction"])
        if overhead > threshold:
            raise StrictValidationError(f"adapter overhead failed: {host}={overhead}")
        hosts[host] = {
            "capsule_files": len(capsule_files),
            "reachable_inventory_rows": len(filesystem_by_path),
            "reachable_regular_file_bytes": int(forbidden_scan["regular_file_bytes"]),
            "roundtrip_rows": len(roundtrips),
            "native_forward_rows": len(forwards),
            "adapter_sha256": expected_adapter,
            "performance_observations": len(alone),
            "overhead_fraction": overhead,
        }
    return {
        "hosts": hosts,
        "hosts_verified": len(hosts),
        "physical_capability_archives_present": sum(
            int(Path(row["path"]).suffix.casefold() in FORBIDDEN_CAPABILITY_SUFFIXES)
            for host in HOSTS
            for row in read_json(
                certification_root / host / "certification-capsule-manifest.json"
            )["files"]
        ),
        "physical_source_success_ledgers_present": sum(
            int("source_success" in str(row["path"]).casefold())
            for host in HOSTS
            for row in read_json(
                certification_root / host / "certification-capsule-manifest.json"
            )["files"]
        ),
    }


def verify_locked_matrix_rows(root: Path, matrix_root: Path | None = None) -> dict[str, Any]:
    """Recompute the full quality/retention matrix without trusting result flags."""

    root = root.resolve()
    matrix_root = matrix_root.resolve() if matrix_root is not None else None
    protocol = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = read_json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    locks = read_json(root / merged["source_success_locks"])
    english_records, domain_records = _matrix_records(root, locks)
    record_maps = {
        "english": {str(row["probe_id"]): row for row in english_records},
        **{
            domain: {str(row["probe_id"]): row for row in domain_records[domain]}
            for domain in DOMAINS
        },
    }
    english_reference, domain_reference = _source_references(root)
    references = {"english": english_reference, **domain_reference}
    expected_keys = {
        (capability, probe_id)
        for capability, records in record_maps.items()
        for probe_id in records
    }
    by_host: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for host in HOSTS:
        path = (
            matrix_root / host / "observations.jsonl"
            if matrix_root is not None
            else root
            / f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/observations.jsonl"
        )
        rows = read_jsonl(path)
        index = {
            (str(row.get("capability")), str(row.get("probe_id"))): row for row in rows
        }
        if len(index) != len(rows) or set(index) != expected_keys:
            raise StrictValidationError(f"full raw matrix row set changed: {host}")
        for key, row in index.items():
            capability, probe_id = key
            output = str(row.get("output"))
            if hashlib.sha256(output.encode("utf-8")).hexdigest() != row.get("output_sha256"):
                raise StrictValidationError(f"matrix output hash changed: {host}/{key}")
            computed_functional = bool(
                evaluate_functional(output, record_maps[capability][probe_id]["evaluator"])
            )
            if not computed_functional:
                raise StrictValidationError(f"matrix functional failure: {host}/{key}")
            if output != references[capability][probe_id]:
                raise StrictValidationError(f"matrix source byte mismatch: {host}/{key}")
            actions = [int(value) for value in row.get("actions", [])]
            if capability != "english" and row.get("actions_sha256") != sha256_bytes(
                canonical_json_bytes(actions)
            ):
                raise StrictValidationError(f"matrix action hash changed: {host}/{key}")
        by_host[host] = index
    cross_output = 0
    cross_actions = 0
    specialist = [key for key in expected_keys if key[0] != "english"]
    for key in expected_keys:
        if len({str(by_host[host][key]["output"]) for host in HOSTS}) != 1:
            raise StrictValidationError(f"cross-host output mismatch: {key}")
        cross_output += 1
    for key in specialist:
        if len(
            {
                tuple(int(value) for value in by_host[host][key].get("actions", []))
                for host in HOSTS
            }
        ) != 1:
            raise StrictValidationError(f"cross-host specialist action mismatch: {key}")
        cross_actions += 1
    return {
        "hosts": len(HOSTS),
        "capabilities": len(CAPABILITIES),
        "rows_per_host": len(expected_keys),
        "rows_verified": len(expected_keys) * len(HOSTS),
        "cross_host_outputs_equal": cross_output,
        "cross_host_specialist_actions_equal": cross_actions,
    }


def verify_live_causality(
    root: Path, causality_root: Path | None = None
) -> dict[str, Any]:
    """Derive causal results from new live raw rows, never prior matrix outputs."""

    root = root.resolve()
    causality_root = (
        (root / CAUSALITY_ROOT).resolve()
        if causality_root is None
        else causality_root.resolve()
    )
    source_path = root / "abi_v2/live_causality.py"
    try:
        transitive_sources = execution_source_manifest(root)
    except ExecutionSourceError as exc:
        raise StrictValidationError(f"transitive execution sources unavailable: {exc}") from exc
    source_text = read_text(source_path)
    for forbidden in ("_matrix_rows", "_matrix_result", "_source_references"):
        if forbidden in source_text:
            raise StrictValidationError(f"live causality source reads replay evidence: {forbidden}")
    protocol = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = read_json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    locks = read_json(root / merged["source_success_locks"])
    adapters = read_json(root / merged["adapter_manifest"])["adapters"]
    english_records, domain_records = _matrix_records(root, locks)
    expected_selected = _selected(english_records, domain_records, 32)
    expected_ids = {
        capability: [str(row["probe_id"]) for row in expected_selected[capability]]
        for capability in CAPABILITIES
    }
    expected_prompt_hashes = {
        (capability, str(row["probe_id"])): hashlib.sha256(
            str(row["prompt"]).encode("utf-8")
        ).hexdigest()
        for capability in CAPABILITIES
        for row in expected_selected[capability]
    }
    current_packages = {
        capability: sha256_file(root / path)
        for capability, path in CAPABILITY_PATHS.items()
    }
    all_rows: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    by_host: dict[str, Any] = {}
    positive_conditions = CONDITIONS[:6]
    for host in HOSTS:
        base_path = causality_root / host
        manifest = read_json(base_path / "manifest.json")
        verify_evidence_hash(manifest, label=f"causality/{host}/manifest")
        if manifest.get("format") != "abi-v2-live-host-causality-run/3":
            raise StrictValidationError(f"live causality format changed: {host}")
        rows = read_jsonl(base_path / "observations.jsonl")
        if manifest.get("observations_sha256") != sha256_file(
            base_path / "observations.jsonl"
        ):
            raise StrictValidationError(f"live causality raw binding changed: {host}")
        if manifest.get("execution_source_sha256") != sha256_file(source_path):
            raise StrictValidationError(f"stale live causality code binding: {host}")
        if manifest.get("transitive_execution_sources") != transitive_sources:
            raise StrictValidationError(f"stale transitive execution source binding: {host}")
        if manifest.get("sample_seed") != SAMPLE_SEED or manifest.get(
            "samples_per_capability"
        ) != 32:
            raise StrictValidationError(f"live causal selection changed: {host}")
        if manifest.get("selected_probe_ids") != expected_ids:
            raise StrictValidationError(f"live causal selected IDs changed: {host}")
        if manifest.get("conditions") != list(CONDITIONS):
            raise StrictValidationError(f"live causal interventions changed: {host}")
        expected_count = len(CAPABILITIES) * 32 * len(CONDITIONS)
        if len(rows) != expected_count or manifest.get("observations_rows") != expected_count:
            raise StrictValidationError(f"live causal raw row depth changed: {host}")
        index = {
            (str(row.get("condition")), str(row.get("capability")), str(row.get("probe_id"))): row
            for row in rows
        }
        if len(index) != len(rows):
            raise StrictValidationError(f"duplicate live causal row: {host}")
        expected_keys = {
            (condition, capability, probe_id)
            for condition in CONDITIONS
            for capability, ids in expected_ids.items()
            for probe_id in ids
        }
        if set(index) != expected_keys:
            raise StrictValidationError(f"live causal row set changed: {host}")
        if any(
            row.get("prompt_sha256")
            != expected_prompt_hashes[(capability, probe_id)]
            for (_, capability, probe_id), row in index.items()
        ):
            raise StrictValidationError(f"live causal prompt binding changed: {host}")

        processes = manifest.get("condition_processes")
        if not isinstance(processes, list) or len(processes) != len(CONDITIONS):
            raise StrictValidationError(f"fresh condition process ledger missing: {host}")
        process_by_condition = {str(row.get("condition")): row for row in processes}
        if set(process_by_condition) != set(CONDITIONS):
            raise StrictValidationError(f"fresh condition process set changed: {host}")
        process_ids = [int(process_by_condition[name].get("process_id", -1)) for name in CONDITIONS]
        if any(value <= 0 for value in process_ids) or len(set(process_ids)) != len(CONDITIONS):
            raise StrictValidationError(f"causal conditions did not use fresh processes: {host}")
        condition_receipts: dict[str, dict[str, Any]] = {}
        for condition in CONDITIONS:
            receipt_path = base_path / "conditions" / f"{condition}.json"
            receipt = read_json(receipt_path)
            verify_evidence_hash(receipt, label=f"causality/{host}/{condition}")
            if (
                receipt.get("format") != "abi-v2-live-host-condition/3"
                or receipt.get("host") != host
                or receipt.get("condition") != condition
                or receipt.get("execution_source_sha256") != sha256_file(source_path)
                or receipt.get("adapter_sha256") != adapters[host]["sha256"]
                or receipt.get("capability_sha256") != current_packages
                or receipt.get("transitive_execution_sources") != transitive_sources
                or int(receipt.get("process_id", -1)) != process_by_condition[condition][
                    "process_id"
                ]
                or sha256_file(receipt_path)
                != process_by_condition[condition]["receipt_sha256"]
            ):
                raise StrictValidationError(f"condition receipt binding changed: {host}/{condition}")
            condition_rows = [row for row in rows if row.get("condition") == condition]
            condition_bytes = b"".join(canonical_json_bytes(row) for row in condition_rows)
            if (
                len(condition_rows) != len(CAPABILITIES) * 32
                or receipt.get("observations_rows") != len(condition_rows)
                or receipt.get("observations_sha256")
                != hashlib.sha256(condition_bytes).hexdigest()
                or process_by_condition[condition].get("observations_sha256")
                != receipt.get("observations_sha256")
            ):
                raise StrictValidationError(f"condition raw rows changed: {host}/{condition}")
            intervention = receipt.get("intervention")
            if not isinstance(intervention, dict):
                raise StrictValidationError(f"condition intervention missing: {host}/{condition}")
            intervention_payload = {
                key: value
                for key, value in intervention.items()
                if key != "intervention_sha256"
            }
            if intervention.get("intervention_sha256") != sha256_bytes(
                canonical_json_bytes(intervention_payload)
            ):
                raise StrictValidationError(f"condition intervention hash changed: {host}/{condition}")
            before = intervention.get("values_before")
            after = intervention.get("values_after")
            if not isinstance(before, list) or not isinstance(after, list):
                raise StrictValidationError(f"condition intervention rows missing: {host}/{condition}")
            if intervention.get("kind") == "live_native_parameter_intervention":
                if (
                    len(before) != len(after)
                    or int(intervention.get("elements_intervened", -1)) != len(before)
                    or intervention.get("before_sha256")
                    != sha256_bytes(canonical_json_bytes(before))
                    or intervention.get("after_sha256")
                    != sha256_bytes(canonical_json_bytes(after))
                    or any(not math.isfinite(float(value)) for value in (*before, *after))
                ):
                    raise StrictValidationError(
                        f"native intervention bytes unrecomputable: {host}/{condition}"
                    )
                expected_seed = (
                    f"{SAMPLE_SEED}:{host}:{condition}:"
                    f"{intervention.get('parameter_name')}"
                )
                if intervention.get("seed") != expected_seed:
                    raise StrictValidationError(f"native intervention seed changed: {host}")
                if condition == "neutral_host" and any(
                    float(value) != 0.5 for value in after
                ):
                    raise StrictValidationError(f"neutral host mutation malformed: {host}")
                if condition == "zero_state" and any(
                    float(value) != 0.0 for value in after
                ):
                    raise StrictValidationError(f"zero-state mutation malformed: {host}")
                if condition == "random_state":
                    generator = random.Random(expected_seed)
                    expected_after = [
                        generator.randrange(-32, 33) / 1024.0
                        for _ in range(len(before))
                    ]
                    if after != expected_after:
                        raise StrictValidationError(f"random-state mutation malformed: {host}")
                if condition == "shuffled_state":
                    expected_after = list(before)
                    random.Random(expected_seed).shuffle(expected_after)
                    if after != expected_after:
                        raise StrictValidationError(f"shuffled-state mutation malformed: {host}")
                should_mutate = condition in {
                    "neutral_host",
                    "zero_state",
                    "random_state",
                    "shuffled_state",
                }
                if should_mutate != (after != before):
                    raise StrictValidationError(f"native intervention was not applied: {host}/{condition}")
            elif host != "layercake" and condition != "host_removed":
                raise StrictValidationError(f"native intervention tensor absent: {host}/{condition}")
            native = receipt.get("native_host")
            if not isinstance(native, dict):
                raise StrictValidationError(f"native execution identity missing: {host}/{condition}")
            if condition == "host_removed":
                if (
                    receipt.get("snapshot_argument") != "absent"
                    or int(native.get("parameter_count", -1)) != 0
                    or native.get("runtime_mode")
                    != "physically_removed_no_snapshot_no_object"
                    or intervention.get("kind") != "structural_native_host_absence"
                    or intervention.get("values_before") != []
                    or intervention.get("values_after") != []
                ):
                    raise StrictValidationError(f"host removal was not physical: {host}")
            elif host != "layercake" and (
                receipt.get("snapshot_argument") != "present"
                or int(native.get("parameter_count", 0)) <= 0
                or native.get("runtime_mode") != "live_transformer_checkpoint"
                or native.get("snapshot_inventory_sha256")
                != merged["host_registry"][host]["snapshot_inventory_sha256"]
                or native.get("checkpoint_sha256")
                != merged["host_registry"][host]["checkpoint_sha256"]
                or native.get("tokenizer_sha256")
                != merged["host_registry"][host]["tokenizer_sha256"]
            ):
                raise StrictValidationError(f"native host did not execute live: {host}/{condition}")
            elif host == "layercake" and (
                receipt.get("snapshot_argument") != "absent"
                or int(native.get("parameter_count", -1)) != 0
                or native.get("runtime_mode") != "capability_native_layercake_host"
            ):
                raise StrictValidationError(f"LayerCake host identity changed: {condition}")
            condition_receipts[condition] = receipt

        for capability, ids in expected_ids.items():
            for probe_id in ids:
                real = index[("real_host", capability, probe_id)]
                for condition in positive_conditions:
                    row = index[(condition, capability, probe_id)]
                    output = row.get("capability_output")
                    realized = row.get("realized_output")
                    if not isinstance(output, str) or not isinstance(realized, str):
                        raise StrictValidationError(
                            f"live positive execution missing: {host}/{condition}/{capability}/{probe_id}"
                        )
                    if hashlib.sha256(output.encode("utf-8")).hexdigest() != row.get(
                        "capability_output_sha256"
                    ):
                        raise StrictValidationError(f"live capability output hash changed: {host}")
                    if hashlib.sha256(realized.encode("utf-8")).hexdigest() != row.get(
                        "realized_output_sha256"
                    ):
                        raise StrictValidationError(f"live realized output hash changed: {host}")
                    state = row.get("host_state")
                    if not isinstance(state, dict) or set(state) != {
                        "condition",
                        "intervention_sha256",
                        "state_vector",
                    }:
                        raise StrictValidationError(f"applied host-state schema changed: {host}")
                    state_vector = state["state_vector"]
                    if not isinstance(state_vector, list) or any(
                        not math.isfinite(float(value)) for value in state_vector
                    ):
                        raise StrictValidationError(f"applied host state invalid: {host}")
                    expected_state_hash = sha256_bytes(canonical_json_bytes(state))
                    if (
                        state.get("condition") != condition
                        or state.get("intervention_sha256")
                        != condition_receipts[condition]["intervention"][
                            "intervention_sha256"
                        ]
                        or row.get("host_state_sha256") != expected_state_hash
                        or row.get("applied_host_state_sha256") != expected_state_hash
                    ):
                        raise StrictValidationError(f"host state was not consumed live: {host}")
                    if host == "layercake":
                        prompt_hash = expected_prompt_hashes[(capability, probe_id)]
                        if condition == "real_host":
                            expected_layercake_state = [1.0]
                        elif condition == "neutral_host":
                            expected_layercake_state = [0.5]
                        elif condition == "zero_state":
                            expected_layercake_state = [0.0]
                        elif condition == "random_state":
                            expected_layercake_state = [
                                random.Random(
                                    f"{SAMPLE_SEED}:{host}:{condition}:{prompt_hash}"
                                ).uniform(-1.0, 1.0)
                            ]
                        elif condition == "shuffled_state":
                            expected_layercake_state = [1.0]
                        else:
                            expected_layercake_state = []
                        if state_vector != expected_layercake_state:
                            raise StrictValidationError(
                                f"LayerCake state intervention changed: {condition}/{probe_id}"
                            )
                    elif condition != "host_removed" and len(state_vector) != 32:
                        raise StrictValidationError(
                            f"native forward state depth changed: {host}/{condition}"
                        )
                    if row.get("actions_sha256") != sha256_bytes(
                        canonical_json_bytes([int(value) for value in row.get("actions", [])])
                    ):
                        raise StrictValidationError(f"live action hash changed: {host}")
                    if realized != real["realized_output"] or output != real["capability_output"]:
                        raise StrictValidationError(
                            f"host-state intervention changed output: {host}/{condition}/{capability}/{probe_id}"
                        )
                    if condition == "host_removed" and state_vector:
                        raise StrictValidationError(f"host-removed state present: {host}")
                adapter_removed = index[("adapter_removed", capability, probe_id)]
                adapter_state = adapter_removed.get("host_state")
                adapter_state_hash = (
                    sha256_bytes(canonical_json_bytes(adapter_state))
                    if isinstance(adapter_state, dict)
                    else None
                )
                if (
                    not isinstance(adapter_removed.get("capability_output"), str)
                    or adapter_removed.get("realized_output") is not None
                    or not adapter_removed.get("exception_type")
                    or not isinstance(adapter_state, dict)
                    or set(adapter_state) != {
                        "condition",
                        "intervention_sha256",
                        "state_vector",
                    }
                    or adapter_state.get("condition") != "adapter_removed"
                    or adapter_state.get("intervention_sha256")
                    != condition_receipts["adapter_removed"]["intervention"][
                        "intervention_sha256"
                    ]
                    or adapter_removed.get("host_state_sha256") != adapter_state_hash
                ):
                    raise StrictValidationError(f"adapter removal did not fail live: {host}")
                if adapter_removed["capability_output"] != real["capability_output"]:
                    raise StrictValidationError(f"adapter-removal generation was not live: {host}")
                capability_removed = index[("capability_removed", capability, probe_id)]
                removed_state = capability_removed.get("host_state")
                removed_state_hash = (
                    sha256_bytes(canonical_json_bytes(removed_state))
                    if isinstance(removed_state, dict)
                    else None
                )
                if (
                    capability_removed.get("capability_output") is not None
                    or capability_removed.get("realized_output") is not None
                    or not capability_removed.get("exception_type")
                    or not isinstance(removed_state, dict)
                    or set(removed_state) != {
                        "condition",
                        "intervention_sha256",
                        "state_vector",
                    }
                    or removed_state.get("condition") != "capability_removed"
                    or removed_state.get("intervention_sha256")
                    != condition_receipts["capability_removed"]["intervention"][
                        "intervention_sha256"
                    ]
                    or capability_removed.get("host_state_sha256") != removed_state_hash
                ):
                    raise StrictValidationError(f"capability removal did not fail live: {host}")
        all_rows[host] = index
        by_host[host] = {
            "raw_rows": len(rows),
            "live_positive_executions": len(CAPABILITIES) * 32 * len(positive_conditions),
            "live_adapter_removals": len(CAPABILITIES) * 32,
            "live_capability_removals": len(CAPABILITIES) * 32,
            "fresh_condition_processes": len(process_ids),
            "applied_native_parameter_interventions": sum(
                int(
                    receipt["intervention"].get("kind")
                    == "live_native_parameter_intervention"
                )
                for receipt in condition_receipts.values()
            ),
        }

    cross_host = 0
    for capability, ids in expected_ids.items():
        for probe_id in ids:
            values = {
                all_rows[host][("real_host", capability, probe_id)]["realized_output"]
                for host in HOSTS
            }
            if len(values) != 1:
                raise StrictValidationError(
                    f"new live real-host outputs differ: {capability}/{probe_id}"
                )
            cross_host += 1
    return {
        "hosts": by_host,
        "raw_rows": sum(row["raw_rows"] for row in by_host.values()),
        "cross_host_real_outputs_equal": cross_host,
        "applied_host_state_channel": "AppliedHostStateAdapter.realize(host_state=...)",
        "causal_conclusion": (
            "Eight fresh processes per host execute the frozen release. Qwen/Pythia neutral, zero, "
            "random, and shuffled conditions mutate a live native parameter tensor and run new "
            "forwards; every positive state is consumed by the conformance adapter while canonical "
            "capability bytes remain invariant. Host removal receives no checkpoint path or native "
            "objects. Adapter removal fails realization and capability removal fails generation."
        ),
    }


def verify_live_isolation(
    root: Path, isolation_root: Path | None = None
) -> dict[str, Any]:
    """Re-evaluate fresh isolation outputs against the frozen functional evaluators."""

    root = root.resolve()
    isolation_root = (
        (root / ISOLATION_ROOT).resolve()
        if isolation_root is None
        else isolation_root.resolve()
    )
    source_path = root / "abi_v2/live_isolation.py"
    try:
        transitive_sources = execution_source_manifest(root)
    except ExecutionSourceError as exc:
        raise StrictValidationError(f"transitive execution sources unavailable: {exc}") from exc
    source_text = read_text(source_path)
    for forbidden in ("_source_references", "_matrix_rows", "_matrix_result"):
        if forbidden in source_text:
            raise StrictValidationError(f"live isolation reads evaluator answers: {forbidden}")
    amendment = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base_protocol = read_json(root / amendment["base_protocol"])
    protocol = {**base_protocol, **amendment}
    locks = read_json(root / protocol["source_success_locks"])
    adapters = read_json(root / protocol["adapter_manifest"])["adapters"]
    english_records, domain_records = _matrix_records(root, locks)
    english_map = {str(row["probe_id"]): row for row in english_records[:100]}
    domain_maps = {
        domain: {str(row["probe_id"]): row for row in domain_records[domain]}
        for domain in DOMAINS
    }
    expected_keys = set()
    for domain in DOMAINS:
        expected_keys.update(
            ("english_only_specialist_target", domain, probe_id)
            for probe_id in domain_maps[domain]
        )
        expected_keys.update(
            ("wrong_specialist_capability", domain, probe_id)
            for probe_id in domain_maps[domain]
        )
    expected_keys.update(
        ("wrong_specialist_on_english", "english", probe_id)
        for probe_id in english_map
    )
    by_host_rows: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    hosts = {}
    for host in HOSTS:
        base_path = isolation_root / host
        manifest = read_json(base_path / "manifest.json")
        verify_evidence_hash(manifest, label=f"isolation/{host}/manifest")
        rows_path = base_path / "observations.jsonl"
        rows = read_jsonl(rows_path)
        if (
            manifest.get("format") != "abi-v2-live-capability-isolation/2"
            or manifest.get("execution_source_sha256") != sha256_file(source_path)
        ):
            raise StrictValidationError(f"stale live isolation source: {host}")
        if manifest.get("transitive_execution_sources") != transitive_sources:
            raise StrictValidationError(f"stale transitive isolation source: {host}")
        if manifest.get("observations_sha256") != sha256_file(rows_path):
            raise StrictValidationError(f"live isolation raw binding changed: {host}")
        if manifest.get("adapter_sha256_before") != adapters[host]["sha256"] or manifest.get(
            "adapter_sha256_after"
        ) != adapters[host]["sha256"]:
            raise StrictValidationError(f"live isolation adapter changed: {host}")
        if manifest.get("english_archive_sha256") != sha256_file(
            root / CAPABILITY_PATHS["english"]
        ):
            raise StrictValidationError(f"live isolation English package changed: {host}")
        if manifest.get("domain_archive_sha256") != {
            domain: sha256_file(root / CAPABILITY_PATHS[domain]) for domain in DOMAINS
        }:
            raise StrictValidationError(f"live isolation domain package changed: {host}")
        index = {
            (str(row.get("mode")), str(row.get("target_capability")), str(row.get("probe_id"))): row
            for row in rows
        }
        if len(rows) != 700 or manifest.get("observations_rows") != 700:
            raise StrictValidationError(f"live isolation row depth changed: {host}")
        if len(index) != len(rows) or set(index) != expected_keys:
            raise StrictValidationError(f"live isolation row set changed: {host}")
        successes = 0
        for key, row in index.items():
            mode, target, probe_id = key
            output = row.get("output")
            if not isinstance(output, str) or hashlib.sha256(output.encode("utf-8")).hexdigest() != row.get(
                "output_sha256"
            ):
                raise StrictValidationError(f"live isolation output hash changed: {host}/{key}")
            record = english_map[probe_id] if target == "english" else domain_maps[target][probe_id]
            if hashlib.sha256(str(record["prompt"]).encode("utf-8")).hexdigest() != row.get(
                "prompt_sha256"
            ):
                raise StrictValidationError(f"live isolation prompt binding changed: {host}/{key}")
            actions = [int(value) for value in row.get("actions", [])]
            if mode != "english_only_specialist_target" and row.get(
                "actions_sha256"
            ) != sha256_bytes(canonical_json_bytes(actions)):
                raise StrictValidationError(f"live isolation action hash changed: {host}/{key}")
            successes += int(bool(evaluate_functional(output, record["evaluator"])))
        if successes != 0:
            raise StrictValidationError(f"live capability isolation failed: {host}={successes}/700")
        by_host_rows[host] = index
        hosts[host] = {"raw_rows": len(rows), "target_successes": successes}
    cross_host = 0
    for key in expected_keys:
        if len({by_host_rows[host][key]["output"] for host in HOSTS}) != 1:
            raise StrictValidationError(f"cross-host isolation output changed: {key}")
        cross_host += 1
    return {
        "hosts": hosts,
        "raw_rows": 700 * len(HOSTS),
        "target_successes": 0,
        "cross_host_outputs_equal": cross_host,
    }


def required_input_manifest(root: Path) -> dict[str, Any]:
    """Bind the strict certificate to every file used to derive its claims."""

    root = root.resolve()
    paths = {
        root / "abi/capability_compiler_phase2_common.py",
        root / "abi/capability_compiler_phase2_teacher.py",
        root / "abi/capability_compiler_phase4_b20_v25_physical_screen.py",
        root / "abi/capability_compiler_phase5_construct_screen.py",
        root / "abi/capability_compiler_phase5_selective_product.py",
        root / "abi_v2/__init__.py",
        root / "abi_v2/canonical.py",
        root / "abi_v2/canonical_spec.json",
        root / "abi_v2/capability_matrix.py",
        root / "abi_v2/certification_pivot_runner.sh",
        root / "abi_v2/conformance_suite.json",
        root / "abi_v2/execution_sources.py",
        root / "abi_v2/final_validation.py",
        root / "abi_v2/host_certification.py",
        root / "abi_v2/isolated_certification.py",
        root / "abi_v2/live_causality.py",
        root / "abi_v2/live_isolation.py",
        root / "abi_v2/matrix_protocol.json",
        root / "abi_v2/matrix_protocol_amendment3.json",
        root / "abi_v2/strict_validation.py",
        root / "external_reproduction/model_download_manifest.json",
        root / "catalogs/capability_compiler_phase1_frozen_v1.json",
        root
        / "evidence/current/segregation/english_and_first_domains_certification_v6.json",
        root
        / "results/abi_capability_compiler_phase4_clarification_route_replication/"
        "B40-seed104729-v927/evaluation/development_outputs.jsonl",
        root
        / "results/abi_capability_compiler_phase6_composition/run_v1032/"
        "seed104729/observations.jsonl",
        root / "results/abi_v2/adapters/manifest.json",
        root.parent
        / "layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py",
    }
    # Bind every transitive Python/runtime source, not merely the immediate
    # verifier entry points. The live receipts carry the same aggregate.
    layercake_root = (root.parent / "layercake_release").resolve()
    for tree, suffixes in (
        (root / "abi", {".py"}),
        (root / "abi_v2", {".py", ".json", ".sh"}),
        (layercake_root / "layercake", {".py"}),
        (layercake_root / "layercake_extensions", {".py"}),
    ):
        if not tree.is_dir():
            raise StrictValidationError(f"required execution source tree missing: {tree}")
        paths.update(
            path
            for path in tree.rglob("*")
            if path.is_file()
            and path.suffix.casefold() in suffixes
            and "__pycache__" not in path.parts
        )
    protocol = read_json(root / "abi_v2/matrix_protocol_amendment3.json")
    base = read_json(root / protocol["base_protocol"])
    merged = {**base, **protocol}
    for field in ("source_success_locks", "adapter_manifest"):
        paths.add(root / str(merged[field]))
    suite = read_json(root / "abi_v2/conformance_suite.json")
    paths.add(root / str(suite["reference_vectors"]["path"]))
    paths.update(root / path for path in CAPABILITY_PATHS.values())
    for host in HOSTS:
        cert = root / CERTIFICATION_ROOT / host
        paths.update(
            {
                cert / "launcher-receipt.json",
                cert / "receipt.json",
                cert / "physical-isolation.json",
                cert / "certification-capsule-manifest.json",
                cert / "mountinfo.txt",
                cert / "reachable-filesystem-inventory.jsonl",
                cert / "certification/result.json",
                cert / "certification/performance.json",
                cert / "certification/adapter.json",
                root
                / f"results/abi_v2/capability_matrix/{MATRIX_DIRS[host]}/observations.jsonl",
                root / CAUSALITY_ROOT / host / "manifest.json",
                root / CAUSALITY_ROOT / host / "observations.jsonl",
                root / ISOLATION_ROOT / host / "manifest.json",
                root / ISOLATION_ROOT / host / "observations.jsonl",
            }
        )
        paths.update(
            root / CAUSALITY_ROOT / host / "conditions" / f"{condition}.json"
            for condition in CONDITIONS
        )
    files = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError:
            layercake_root = (root.parent / "layercake_release").resolve()
            try:
                relative = "@layercake_release/" + path.resolve().relative_to(
                    layercake_root
                ).as_posix()
            except ValueError as exc:
                raise StrictValidationError(
                    f"required input escaped release roots: {path}"
                ) from exc
        if not path.is_file():
            raise StrictValidationError(f"required file missing: {path}")
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise StrictValidationError(f"required file unreadable: {path}") from exc
        files.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "files": files,
        "file_count": len(files),
        "aggregate_sha256": sha256_bytes(canonical_json_bytes(files)),
    }


def verify(root: Path) -> dict[str, Any]:
    try:
        certification = verify_certifications(root)
        matrix = verify_locked_matrix_rows(root)
        causality = verify_live_causality(root)
        isolation = verify_live_isolation(root)
        return {
            "format": "abi-v2-strict-final-validation/4",
            "status": "PASS_STRICT_RAW_RECOMPUTATION",
            "certification": certification,
            "locked_matrix": matrix,
            "live_causality": causality,
            "live_isolation": isolation,
            "required_inputs": required_input_manifest(root),
            "trusted_scientific_booleans_consumed": 0,
        }
    except StrictValidationError:
        raise
    except Exception as exc:
        raise StrictValidationError(
            f"required scientific claim is unrecomputable: {type(exc).__name__}: {exc}"
        ) from exc


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        value = verify(Path(args.root))
        value["evidence_sha256"] = evidence_hash(value)
        if args.output:
            path = Path(args.root).resolve() / args.output
            if path.exists():
                raise StrictValidationError(f"immutable strict verifier output exists: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            )
    except Exception as exc:
        error = exc if isinstance(exc, StrictValidationError) else StrictValidationError(
            f"verifier failed closed: {type(exc).__name__}: {exc}"
        )
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
