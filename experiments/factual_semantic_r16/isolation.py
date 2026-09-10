"""Build and execute a physical R16 factual-extraction capsule."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import sha256_file
from experiments.preexisting_representation_r15b.public_qualification import canonical_json_bytes


class R16IsolationError(RuntimeError):
    """Raised when physical R16 isolation cannot be established."""


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise R16IsolationError(f"immutable R16 output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    if len(drive) != 1:
        raise R16IsolationError(f"cannot map R16 path into WSL: {resolved}")
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}"


def build_capsule(root: Path, source_bundle: Path, capsule: Path) -> dict[str, Any]:
    if capsule.exists():
        raise R16IsolationError(f"R16 capsule already exists: {capsule}")
    capsule.mkdir(parents=True)
    sources = {
        "isolated_worker.py": root / "experiments/factual_semantic_r16/isolated_worker.py",
        "source_bundle.json": source_bundle,
    }
    for name, source in sources.items():
        if not source.is_file():
            raise R16IsolationError(f"required R16 input missing: {source}")
        shutil.copyfile(source, capsule / name)
    spec = {
        "format": "abi-r16-generic-factual-compiler/1",
        "views_per_fact": 3,
        "candidate_budget": 12,
        "namespaces": ["chemistry/periodic-table", "geography/national-capitals"],
        "oracle_fields": 0,
        "fact_ids": 0,
        "success_ids": 0,
    }
    spec["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
    _write_json_once(capsule / "spec.json", spec)
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(capsule.iterdir(), key=lambda item: item.name)
    ]
    manifest = {
        "format": "abi-r16-isolated-factual-capsule/1",
        "files": files,
        "reveal_files_included": 0,
        "oracle_fields_included": 0,
        "fact_ids_included": 0,
        "success_ids_included": 0,
        "network": False,
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_json_once(capsule / "manifest.json", manifest)
    return manifest


def run_wsl_isolated_extraction(
    root: Path, source_bundle: Path, destination: Path, distribution: str = "Ubuntu"
) -> dict[str, Any]:
    root = root.resolve()
    destination = destination.resolve()
    if os.name != "nt" or destination.exists():
        raise R16IsolationError("registered Windows/WSL destination contract changed")
    with tempfile.TemporaryDirectory(prefix="abi-r16-stage-") as raw:
        capsule = Path(raw) / "capsule"
        manifest = build_capsule(root, source_bundle.resolve(), capsule)
        linux_capsule = f"/tmp/abi-r16-extraction-{uuid.uuid4().hex}"
        linux_runner = f"/tmp/abi-r16-runner-{uuid.uuid4().hex}.sh"
        sandbox_root = f"/tmp/abi-r16-root-{uuid.uuid4().hex}"
        command = (
            "set -euo pipefail; "
            f"cp -aL '{_wsl_path(capsule)}' '{linux_capsule}'; "
            f"cp -L '{_wsl_path(root / 'experiments/factual_semantic_r16/extraction_pivot_runner.sh')}' '{linux_runner}'; "
            f"chmod 500 '{linux_runner}'; "
            f"ABI_CAPSULE_PATH='{linux_capsule}' ABI_SANDBOX_ROOT='{sandbox_root}' "
            "unshare --mount --pid --fork --ipc --uts --net --propagation private "
            f"'{linux_runner}'; "
            f"mkdir -p '{_wsl_path(destination)}'; "
            f"cp -a '{linux_capsule}/output/.' '{_wsl_path(destination)}/'; "
            f"cp -a '{linux_capsule}/manifest.json' '{_wsl_path(destination)}/manifest.json'"
        )
        completed = subprocess.run(
            ["wsl.exe", "-d", distribution, "-u", "root", "--", "bash", "-lc", command],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise R16IsolationError(
                f"physical R16 extraction failed ({completed.returncode}): {completed.stderr[-4000:]}"
            )
        result_path = destination / "result.json"
        if not result_path.is_file():
            raise R16IsolationError("physical R16 extraction result missing")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        launcher = {
            "format": "abi-r16-isolated-extraction-launcher/1",
            "distribution": distribution,
            "worker_exit_code": completed.returncode,
            "sandbox_policy": "linux-pivot-root-no-network/1",
            "manifest_evidence_sha256": manifest["evidence_sha256"],
            "result_sha256": sha256_file(result_path),
            "mountinfo_sha256": sha256_file(destination / "mountinfo.txt"),
        }
        launcher["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(launcher)).hexdigest()
        _write_json_once(destination / "launcher.json", launcher)
        return {"result": result, "launcher": launcher}
