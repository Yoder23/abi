"""Build and execute an R15A weight-delta-only extraction capsule."""

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
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes


class R15IsolationError(RuntimeError):
    """Raised when physical R15A extraction isolation cannot be proved."""


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise R15IsolationError(f"immutable R15 isolation output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    if len(drive) != 1:
        raise R15IsolationError(f"cannot map R15 path into WSL: {resolved}")
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}"


def build_capsule(
    *,
    worker: Path,
    frontend: Path,
    frontend_spec: dict[str, Any],
    delta: Path,
    capsule: Path,
) -> dict[str, Any]:
    if capsule.exists():
        raise R15IsolationError(f"R15 capsule already exists: {capsule}")
    capsule.mkdir(parents=True)
    inputs = {
        "isolated_worker.py": worker,
        "frontend.safetensors": frontend,
        "delta.safetensors": delta,
    }
    for name, source in inputs.items():
        if not source.is_file():
            raise R15IsolationError(f"required R15 capsule input missing: {source}")
        shutil.copyfile(source, capsule / name)
    _write_json_once(capsule / "frontend_spec.json", frontend_spec)
    rows = [
        {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(capsule.iterdir(), key=lambda item: item.name)
    ]
    manifest = {
        "format": "abi-r15a-isolated-extraction-capsule/1",
        "files": rows,
        "input_classifications": {
            "delta.safetensors": "anonymous_foreign_weight_delta",
            "frontend.safetensors": "frozen_generic_frontend",
            "frontend_spec.json": "frozen_generic_frontend_specification",
            "isolated_worker.py": "generic_extractor_code",
        },
        "capability_reveals_included": 0,
        "operation_tables_included": 0,
        "behavior_rows_included": 0,
        "answers_included": 0,
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_json_once(capsule / "manifest.json", manifest)
    return manifest


def run_wsl_isolated_extraction(
    root: Path,
    *,
    frontend: Path,
    frontend_spec: dict[str, Any],
    delta: Path,
    destination: Path,
    distribution: str = "Ubuntu",
) -> dict[str, Any]:
    """Decode one anonymous delta after pivoting away from the development tree."""
    root = root.resolve()
    destination = destination.resolve()
    if os.name != "nt":
        raise R15IsolationError("this launcher is the registered Windows/WSL path")
    if destination.exists():
        raise R15IsolationError(f"immutable R15 extraction exists: {destination}")
    with tempfile.TemporaryDirectory(prefix="abi-r15-extraction-stage-") as raw:
        staging = Path(raw)
        capsule = staging / "capsule"
        manifest = build_capsule(
            worker=root / "experiments/foreign_neural_state_r15/isolated_worker.py",
            frontend=frontend.resolve(),
            frontend_spec=frontend_spec,
            delta=delta.resolve(),
            capsule=capsule,
        )
        linux_capsule = f"/tmp/abi-r15-extraction-{uuid.uuid4().hex}"
        linux_runner = f"/tmp/abi-r15-extraction-runner-{uuid.uuid4().hex}.sh"
        sandbox_root = f"/tmp/abi-r15-extraction-root-{uuid.uuid4().hex}"
        capsule_source = _wsl_path(capsule)
        destination_target = _wsl_path(destination)
        runner_source = _wsl_path(
            root / "experiments/foreign_neural_state_r15/extraction_pivot_runner.sh"
        )
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
                    f"cp -L '{runner_source}' '{linux_runner}'; "
                    f"chmod 500 '{linux_runner}'; "
                    f"ABI_CAPSULE_PATH='{linux_capsule}' "
                    f"ABI_SANDBOX_ROOT='{sandbox_root}' "
                    "unshare --mount --pid --fork --ipc --uts --net "
                    "--propagation private "
                    f"'{linux_runner}'; "
                    f"mkdir -p '{destination_target}'; "
                    f"cp -a '{linux_capsule}/output/.' '{destination_target}/'; "
                    f"cp -a '{linux_capsule}/manifest.json' '{destination_target}/manifest.json'"
                ),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise R15IsolationError(
                "physical R15 extraction failed "
                f"({completed.returncode}): {completed.stderr[-4000:]}"
            )
        result_path = destination / "result.json"
        if not result_path.is_file():
            raise R15IsolationError("physical R15 extraction result missing")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        launcher = {
            "format": "abi-r15a-isolated-extraction-launcher/1",
            "distribution": distribution,
            "worker_exit_code": completed.returncode,
            "sandbox_policy": "linux-pivot-root-no-network/1",
            "manifest_evidence_sha256": manifest["evidence_sha256"],
            "result_sha256": sha256_file(result_path),
            "mountinfo_sha256": sha256_file(destination / "mountinfo.txt"),
        }
        launcher["evidence_sha256"] = hashlib.sha256(
            canonical_json_bytes(launcher)
        ).hexdigest()
        _write_json_once(destination / "launcher.json", launcher)
        return {"result": result, "launcher": launcher}
