"""Build and execute a physically isolated ABI host-certification capsule.

The certification worker is intentionally incapable of seeing capability
packages, source-success locks, matrix observations, or development evidence.
The parent process builds an allowlisted capsule containing only the canonical
ABI, the generic conformance corpus, certification code, and one host.  The
worker executes after ``pivot_root`` into a tmpfs filesystem containing only
that capsule and read-only runtime dependencies.  The old root, Windows
mounts, network namespace, and every development path are unreachable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, sha256_bytes

CAPSULE_FORMAT = "abi-v2-physical-certification-capsule/1"
ISOLATION_FORMAT = "abi-v2-physical-certification-isolation/1"
FORBIDDEN_SUFFIXES = {".abi", ".cake", ".abix", ".abicir"}
ALLOWED_CLASSIFICATIONS = {
    "abi_specification",
    "adapter_certification_code",
    "generic_certification_corpus",
    "host_code",
    "host_checkpoint",
}


class IsolatedCertificationError(RuntimeError):
    """Raised when certification is not physically capability-blind."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def _copy(
    records: list[dict[str, Any]],
    *,
    source: Path,
    capsule: Path,
    relative: str,
    classification: str,
) -> None:
    if classification not in ALLOWED_CLASSIFICATIONS:
        raise IsolatedCertificationError(f"invalid capsule classification: {classification}")
    if not source.is_file():
        raise IsolatedCertificationError(f"required capsule input missing: {source}")
    target = capsule / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target, follow_symlinks=True)
    records.append(
        {
            "path": Path(relative).as_posix(),
            "classification": classification,
            "bytes": target.stat().st_size,
            "sha256": _sha256_file(target),
        }
    )


def build_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
) -> dict[str, Any]:
    """Create one immutable, allowlisted certification filesystem."""

    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise IsolatedCertificationError(f"capsule destination already exists: {destination}")
    if host_key not in {"layercake", "qwen2", "pythia"}:
        raise IsolatedCertificationError(f"unsupported host: {host_key}")
    destination.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    code = (
        "abi_v2/__init__.py",
        "abi_v2/canonical.py",
        "abi_v2/certification_pivot_runner.sh",
        "abi_v2/host_certification.py",
        "abi_v2/isolated_certification.py",
    )
    for relative in code:
        _copy(
            records,
            source=root / relative,
            capsule=destination,
            relative=f"abi_release/{relative}",
            classification="adapter_certification_code",
        )
    for relative in ("abi_v2/canonical_spec.json", "abi_v2/conformance_suite.json"):
        _copy(
            records,
            source=root / relative,
            capsule=destination,
            relative=f"abi_release/{relative}",
            classification="abi_specification",
        )
    _copy(
        records,
        source=root / "abi_v2/reference_vectors/core_vectors.json",
        capsule=destination,
        relative="abi_release/abi_v2/reference_vectors/core_vectors.json",
        classification="generic_certification_corpus",
    )
    if host_key == "layercake":
        _copy(
            records,
            source=(
                root.parent
                / "layercake_release/layercake_extensions/route_isolated_clarification_core_v25.py"
            ),
            capsule=destination,
            relative=(
                "layercake_release/layercake_extensions/"
                "route_isolated_clarification_core_v25.py"
            ),
            classification="host_code",
        )
    else:
        if snapshot is None or not snapshot.resolve().is_dir():
            raise IsolatedCertificationError(f"{host_key} requires an exact snapshot")
        for source in sorted(snapshot.resolve().iterdir(), key=lambda value: value.name):
            if source.is_file():
                _copy(
                    records,
                    source=source,
                    capsule=destination,
                    relative=f"abi_release/host_snapshot/{source.name}",
                    classification="host_checkpoint",
                )
    if any(Path(row["path"]).suffix.casefold() in FORBIDDEN_SUFFIXES for row in records):
        raise IsolatedCertificationError("capability archive entered certification capsule")
    if any("source_success" in str(row["path"]).casefold() for row in records):
        raise IsolatedCertificationError("source-success data entered certification capsule")
    manifest: dict[str, Any] = {
        "format": CAPSULE_FORMAT,
        "host_key": host_key,
        "files": sorted(records, key=lambda row: row["path"]),
        "allowed_classifications": sorted(ALLOWED_CLASSIFICATIONS),
        "capability_archives_included": 0,
        "source_success_ledgers_included": 0,
    }
    manifest["evidence_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    _write_json(destination / "abi_release/certification-capsule-manifest.json", manifest)
    return manifest


def _capsule_inventory(capsule: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(capsule.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file():
            continue
        relative = path.relative_to(capsule).as_posix()
        if relative == "abi_release/certification-capsule-manifest.json":
            continue
        if relative.startswith("abi_release/output/"):
            continue
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def verify_capsule(capsule: Path) -> dict[str, Any]:
    capsule = capsule.resolve()
    manifest_path = capsule / "abi_release/certification-capsule-manifest.json"
    if not manifest_path.is_file():
        raise IsolatedCertificationError("certification capsule manifest missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != CAPSULE_FORMAT:
        raise IsolatedCertificationError("certification capsule format changed")
    expected_hash = manifest.get("evidence_sha256")
    payload = {key: value for key, value in manifest.items() if key != "evidence_sha256"}
    if expected_hash != sha256_bytes(canonical_json_bytes(payload)):
        raise IsolatedCertificationError("certification capsule manifest hash changed")
    if manifest.get("allowed_classifications") != sorted(ALLOWED_CLASSIFICATIONS):
        raise IsolatedCertificationError("certification capsule classifications changed")
    expected = {
        row["path"]: {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        for row in manifest.get("files", [])
        if row.get("classification") in ALLOWED_CLASSIFICATIONS
    }
    if len(expected) != len(manifest.get("files", [])):
        raise IsolatedCertificationError("unclassified file in certification capsule")
    actual_rows = _capsule_inventory(capsule)
    actual = {
        row["path"]: {"bytes": int(row["bytes"]), "sha256": str(row["sha256"])}
        for row in actual_rows
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            path for path in set(actual) & set(expected) if actual[path] != expected[path]
        )
        raise IsolatedCertificationError(
            f"certification capsule inventory changed: missing={missing}, extra={extra}, changed={changed}"
        )
    capability_archives = [
        row["path"]
        for row in actual_rows
        if Path(row["path"]).suffix.casefold() in FORBIDDEN_SUFFIXES
    ]
    source_success_ledgers = [
        row["path"] for row in actual_rows if "source_success" in row["path"].casefold()
    ]
    forbidden = [
        row["path"]
        for row in actual_rows
        if Path(row["path"]).suffix.casefold() in FORBIDDEN_SUFFIXES
        or "source_success" in row["path"].casefold()
    ]
    if forbidden:
        raise IsolatedCertificationError(f"forbidden certification payload present: {forbidden}")
    return {
        "manifest_sha256": _sha256_file(manifest_path),
        "files_verified": len(actual_rows),
        "inventory_sha256": sha256_bytes(canonical_json_bytes(actual_rows)),
        "capability_archives_present": len(capability_archives),
        "source_success_ledgers_present": len(source_success_ledgers),
        "inventory": actual_rows,
    }


def _decode_mount_path(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _mount_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if "-" not in fields or len(fields) < 10:
            raise IsolatedCertificationError("malformed kernel mount record")
        separator = fields.index("-")
        if separator + 2 >= len(fields):
            raise IsolatedCertificationError("incomplete kernel mount record")
        rows.append(
            {
                "mount_point": _decode_mount_path(fields[4]),
                "mount_options": fields[5].split(","),
                "filesystem_type": fields[separator + 1],
                "source": fields[separator + 2],
                "super_options": fields[separator + 3].split(",")
                if separator + 3 < len(fields)
                else [],
            }
        )
    return rows


def _mount_state() -> dict[str, Any]:
    mountinfo = Path("/proc/self/mountinfo")
    text = mountinfo.read_text(encoding="utf-8") if mountinfo.is_file() else ""
    rows = _mount_rows(text)
    runtime_site = os.environ.get("PYTHONPATH", "")
    allowed_exact = {
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
    unexpected = sorted(
        row["mount_point"]
        for row in rows
        if row["mount_point"] not in allowed_exact
    )
    required = {"/", "/capsule", "/dev", "/etc", "/proc", "/usr", runtime_site}
    present = {row["mount_point"] for row in rows}
    root_rows = [row for row in rows if row["mount_point"] == "/"]
    capsule_rows = [row for row in rows if row["mount_point"] == "/capsule"]
    if (
        os.environ.get("ABI_CERTIFICATION_PIVOT_ROOT") != "1"
        or unexpected
        or required - present
        or len(root_rows) != 1
        or root_rows[0]["filesystem_type"] != "tmpfs"
        or len(capsule_rows) != 1
        or Path("/oldroot").exists()
        or Path("/mnt/c").exists()
    ):
        raise IsolatedCertificationError(
            "worker filesystem is not the exact pivot-root certification sandbox"
        )
    return {
        "platform": sys.platform,
        "mountinfo_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "sandbox_policy": "abi-certification-pivot-root/1",
        "runtime_site_mount": runtime_site,
        "allowed_mount_points": sorted(allowed_exact),
        "mounts": rows,
        "unexpected_mount_points": unexpected,
        "old_root_present": Path("/oldroot").exists(),
        "windows_mount_present": Path("/mnt/c").exists(),
    }


def _physical_forbidden_scan() -> dict[str, Any]:
    """Inventory forbidden campaign payload names across the reachable root."""

    forbidden_archives: list[str] = []
    source_success_ledgers: list[str] = []
    files_scanned = 0
    directories_scanned = 0
    excluded_virtual_roots = {"/dev", "/proc"}
    for directory, names, filenames in os.walk("/", topdown=True, followlinks=False):
        directory_path = Path(directory)
        names[:] = [
            name
            for name in names
            if (directory_path / name).as_posix() not in excluded_virtual_roots
        ]
        directories_scanned += 1
        for name in filenames:
            path = directory_path / name
            if path.is_symlink():
                continue
            files_scanned += 1
            normalized = path.as_posix()
            if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
                forbidden_archives.append(normalized)
            if "source_success" in normalized.casefold():
                source_success_ledgers.append(normalized)
    if forbidden_archives or source_success_ledgers:
        raise IsolatedCertificationError(
            "forbidden campaign payload is reachable inside certification root"
        )
    return {
        "scan_root": "/",
        "excluded_virtual_roots": sorted(excluded_virtual_roots),
        "directories_scanned": directories_scanned,
        "files_scanned": files_scanned,
        "capability_archive_paths": forbidden_archives,
        "source_success_ledger_paths": source_success_ledgers,
    }


def run_worker(capsule: Path, *, host_key: str, device: str) -> dict[str, Any]:
    """Execute certification after proving capsule identity and mount isolation."""

    capsule = capsule.resolve()
    root = capsule / "abi_release"
    inventory = verify_capsule(capsule)
    mount = _mount_state()
    forbidden_scan = _physical_forbidden_scan()
    if sys.platform != "linux" or mount["unexpected_mount_points"]:
        raise IsolatedCertificationError(
            "worker requires the exact Linux pivot-root isolation policy"
        )
    from .host_certification import certify_host

    output = root / "output"
    snapshot = root / "host_snapshot" if host_key != "layercake" else None
    isolation = {
        "format": ISOLATION_FORMAT,
        "host_key": host_key,
        "capsule": inventory,
        "mount": mount,
        "reachable_filesystem_forbidden_scan": forbidden_scan,
        "process_id": os.getpid(),
        "parent_process_id": os.getppid(),
        "capsule_files_physically_present": inventory["files_verified"],
        "capability_archives_physically_present": inventory[
            "capability_archives_present"
        ],
        "source_success_ledgers_physically_present": inventory[
            "source_success_ledgers_present"
        ],
    }
    isolation["evidence_sha256"] = sha256_bytes(canonical_json_bytes(isolation))
    result = certify_host(
        root,
        host_key=host_key,
        output_dir=output / "certification",
        snapshot=snapshot,
        device=device,
        physical_isolation=isolation,
    )
    _write_json(output / "physical-isolation.json", isolation)
    shutil.copy2(
        root / "certification-capsule-manifest.json",
        output / "certification-capsule-manifest.json",
    )
    mountinfo = Path("/proc/self/mountinfo")
    mountinfo_bytes = mountinfo.read_bytes() if mountinfo.is_file() else b""
    (output / "mountinfo.txt").write_bytes(mountinfo_bytes)
    receipt: dict[str, Any] = {
        "format": "abi-v2-isolated-host-certification-receipt/1",
        "host_key": host_key,
        "device": device,
        "isolation_evidence_sha256": _sha256_file(output / "physical-isolation.json"),
        "result_sha256": _sha256_file(output / "certification/result.json"),
        "adapter_sha256": _sha256_file(output / "certification/adapter.json"),
        "performance_sha256": _sha256_file(output / "certification/performance.json"),
        "capsule_manifest_sha256": _sha256_file(
            output / "certification-capsule-manifest.json"
        ),
        "mountinfo_sha256": _sha256_file(output / "mountinfo.txt"),
        "executed_isolated_certification_sha256": _sha256_file(
            root / "abi_v2/isolated_certification.py"
        ),
        "executed_pivot_runner_sha256": _sha256_file(
            root / "abi_v2/certification_pivot_runner.sh"
        ),
        "executed_host_certification_sha256": _sha256_file(
            root / "abi_v2/host_certification.py"
        ),
        "worker_exit_basis": result["status"],
    }
    receipt["evidence_sha256"] = sha256_bytes(canonical_json_bytes(receipt))
    _write_json(output / "receipt.json", receipt)
    return receipt


def run_wsl_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
    device: str = "cpu",
    distribution: str = "Ubuntu",
) -> dict[str, Any]:
    """Run a capsule in WSL2 after detaching the Windows development drive."""

    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise IsolatedCertificationError(f"immutable output exists: {destination}")
    capsule = Path(os.environ.get("TEMP", str(root))) / (
        f"abi-certification-{host_key}-{uuid.uuid4().hex}"
    )
    build_capsule(root, host_key=host_key, destination=capsule, snapshot=snapshot)

    def wsl_path(path: Path) -> str:
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").casefold()
        if len(drive) != 1:
            raise IsolatedCertificationError(f"cannot map path into WSL: {resolved}")
        tail = resolved.as_posix().split(":", 1)[1]
        return f"/mnt/{drive}{tail}"

    capsule_source = wsl_path(capsule)
    destination_target = wsl_path(destination)
    runtime_probe = subprocess.run(
        [
            "wsl.exe",
            "-d",
            distribution,
            "--",
            "python3",
            "-c",
            "import site; print(site.getusersitepackages())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    runtime_site = runtime_probe.stdout.strip()
    if runtime_probe.returncode != 0 or not runtime_site.startswith("/home/"):
        raise IsolatedCertificationError("cannot identify the WSL Python runtime site")
    linux_capsule = f"/tmp/abi-certification-{host_key}-{uuid.uuid4().hex}"
    sandbox_root = f"/tmp/abi-certification-root-{host_key}-{uuid.uuid4().hex}"
    completed = subprocess.run(
        [
            "wsl.exe",
            "-d",
            distribution,
            "-u",
            "root",
            "--",
            "bash",
            "-lc",
            (
                "set -euo pipefail; "
                f"cp -aL '{capsule_source}' '{linux_capsule}'; "
                f"chmod 500 '{linux_capsule}/abi_release/abi_v2/certification_pivot_runner.sh'; "
                f"ABI_CAPSULE_PATH='{linux_capsule}' "
                f"ABI_SANDBOX_ROOT='{sandbox_root}' "
                f"ABI_HOST_KEY='{host_key}' ABI_DEVICE='{device}' "
                f"ABI_RUNTIME_SITE_PATH='{runtime_site}' "
                "unshare --mount --pid --fork --ipc --uts --net "
                "--propagation private "
                f"'{linux_capsule}/abi_release/abi_v2/certification_pivot_runner.sh'; "
                f"mkdir -p '{Path(destination_target).parent.as_posix()}'; "
                f"cp -a '{linux_capsule}/abi_release/output' '{destination_target}'; "
                f"printf '%s' '{linux_capsule}'"
            ),
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedCertificationError(
            "WSL isolated certification failed "
            f"({completed.returncode}): {completed.stderr[-4000:]}"
        )
    receipt_path = destination / "receipt.json"
    if not receipt_path.is_file():
        raise IsolatedCertificationError("isolated certification receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["launcher"] = {
        "distribution": distribution,
        "exit_code": completed.returncode,
        "sandbox_policy": "abi-certification-pivot-root/1",
        "wsl_capsule_path_sha256": hashlib.sha256(
            completed.stdout.strip().encode("utf-8")
        ).hexdigest(),
        "windows_capsule_path_sha256": hashlib.sha256(
            str(capsule).encode("utf-8")
        ).hexdigest(),
    }
    receipt["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "evidence_sha256"})
    )
    _write_json(destination / "launcher-receipt.json", receipt)
    return receipt


def run_linux_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a capsule in a private Linux mount namespace."""

    root = root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise IsolatedCertificationError(f"immutable output exists: {destination}")
    if root == Path(root.anchor):
        raise IsolatedCertificationError("refusing to mask a filesystem root")
    staging = Path(tempfile.mkdtemp(prefix=f"abi-certification-{host_key}-"))
    capsule = staging / "capsule"
    build_capsule(root, host_key=host_key, destination=capsule, snapshot=snapshot)
    runner = capsule / "abi_release/abi_v2/certification_pivot_runner.sh"
    runner.chmod(0o500)
    sandbox_root = f"/tmp/abi-certification-root-{host_key}-{uuid.uuid4().hex}"
    environment = {
        **os.environ,
        "ABI_CAPSULE_PATH": capsule.as_posix(),
        "ABI_SANDBOX_ROOT": sandbox_root,
        "ABI_HOST_KEY": host_key,
        "ABI_DEVICE": device,
        "ABI_RUNTIME_SITE_PATH": site.getusersitepackages(),
    }
    completed = subprocess.run(
        [
            "unshare",
            "--user",
            "--map-root-user",
            "--mount",
            "--pid",
            "--fork",
            "--ipc",
            "--uts",
            "--net",
            "--propagation",
            "private",
            str(runner),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise IsolatedCertificationError(
            "Linux isolated certification failed "
            f"({completed.returncode}): {completed.stderr[-4000:]}"
        )
    shutil.copytree(capsule / "abi_release/output", destination)
    receipt = json.loads((destination / "receipt.json").read_text(encoding="utf-8"))
    receipt["launcher"] = {
        "kind": "linux-user-mount-namespace",
        "exit_code": completed.returncode,
        "capsule_path_sha256": hashlib.sha256(str(capsule).encode("utf-8")).hexdigest(),
        "masked_release_root_sha256": hashlib.sha256(str(root).encode("utf-8")).hexdigest(),
    }
    receipt["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes({key: value for key, value in receipt.items() if key != "evidence_sha256"})
    )
    _write_json(destination / "launcher-receipt.json", receipt)
    return receipt


def run_isolated_capsule(
    root: Path,
    *,
    host_key: str,
    destination: Path,
    snapshot: Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    if os.name == "nt":
        return run_wsl_capsule(
            root,
            host_key=host_key,
            destination=destination,
            snapshot=snapshot,
            device=device,
        )
    if sys.platform == "linux":
        return run_linux_capsule(
            root,
            host_key=host_key,
            destination=destination,
            snapshot=snapshot,
            device=device,
        )
    raise IsolatedCertificationError("no supported physical certification sandbox is available")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    build_parser.add_argument("--destination", required=True)
    build_parser.add_argument("--snapshot")
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--capsule", required=True)
    worker_parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    worker_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    run_parser = subparsers.add_parser("run-wsl")
    run_parser.add_argument("--host", required=True, choices=("layercake", "qwen2", "pythia"))
    run_parser.add_argument("--destination", required=True)
    run_parser.add_argument("--snapshot")
    run_parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    if args.command == "build":
        value = build_capsule(
            Path.cwd(),
            host_key=args.host,
            destination=Path(args.destination),
            snapshot=Path(args.snapshot) if args.snapshot else None,
        )
    elif args.command == "worker":
        value = run_worker(Path(args.capsule), host_key=args.host, device=args.device)
    else:
        value = run_wsl_capsule(
            Path.cwd(),
            host_key=args.host,
            destination=Path(args.destination),
            snapshot=Path(args.snapshot) if args.snapshot else None,
            device=args.device,
        )
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
