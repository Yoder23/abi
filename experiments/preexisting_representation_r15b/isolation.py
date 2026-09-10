"""Build and execute a physically isolated R15B representation capsule."""

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

from .public_qualification import canonical_json_bytes


class R15BIsolationError(RuntimeError):
    """Raised when physical R15B isolation cannot be established."""


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise R15BIsolationError(f"immutable R15B output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def _wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").casefold()
    if len(drive) != 1:
        raise R15BIsolationError(f"cannot map R15B path into WSL: {resolved}")
    return f"/mnt/{drive}{resolved.as_posix().split(':', 1)[1]}"


def build_capsule(*, root: Path, representation: Path, capsule: Path) -> dict[str, Any]:
    if capsule.exists():
        raise R15BIsolationError(f"R15B capsule already exists: {capsule}")
    capsule.mkdir(parents=True)
    sources = {
        "isolated_worker.py": root
        / "experiments/preexisting_representation_r15b/isolated_worker.py",
        "representation.safetensors": representation,
    }
    for name, source in sources.items():
        if not source.is_file():
            raise R15BIsolationError(f"required R15B input missing: {source}")
        shutil.copyfile(source, capsule / name)
    spec = {
        "format": "abi-r15b-generic-representation-decoder/1",
        "input": "six_anonymous_pre_answer_residuals_plus_eight_output_rows",
        "prompts": 0,
        "answers": 0,
        "semantic_labels": 0,
        "candidate_search": False,
    }
    spec["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(spec)).hexdigest()
    _write_json_once(capsule / "spec.json", spec)
    files = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(capsule.iterdir(), key=lambda item: item.name)
    ]
    manifest = {
        "format": "abi-r15b-isolated-representation-capsule/1",
        "files": files,
        "reveal_files_included": 0,
        "prompts_included": 0,
        "answers_included": 0,
        "semantic_labels_included": 0,
        "network": False,
    }
    manifest["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    _write_json_once(capsule / "manifest.json", manifest)
    return manifest


def run_wsl_isolated_extraction(
    root: Path,
    *,
    representation: Path,
    destination: Path,
    distribution: str = "Ubuntu",
) -> dict[str, Any]:
    root = root.resolve()
    destination = destination.resolve()
    if os.name != "nt" or destination.exists():
        raise R15BIsolationError("registered Windows/WSL destination contract changed")
    with tempfile.TemporaryDirectory(prefix="abi-r15b-stage-") as raw:
        capsule = Path(raw) / "capsule"
        manifest = build_capsule(
            root=root, representation=representation.resolve(), capsule=capsule
        )
        linux_capsule = f"/tmp/abi-r15b-extraction-{uuid.uuid4().hex}"
        linux_runner = f"/tmp/abi-r15b-runner-{uuid.uuid4().hex}.sh"
        sandbox_root = f"/tmp/abi-r15b-root-{uuid.uuid4().hex}"
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
                    f"cp -aL '{_wsl_path(capsule)}' '{linux_capsule}'; "
                    f"cp -L '{_wsl_path(root / 'experiments/preexisting_representation_r15b/extraction_pivot_runner.sh')}' '{linux_runner}'; "
                    f"chmod 500 '{linux_runner}'; "
                    f"ABI_CAPSULE_PATH='{linux_capsule}' ABI_SANDBOX_ROOT='{sandbox_root}' "
                    "unshare --mount --pid --fork --ipc --uts --net --propagation private "
                    f"'{linux_runner}'; "
                    f"mkdir -p '{_wsl_path(destination)}'; "
                    f"cp -a '{linux_capsule}/output/.' '{_wsl_path(destination)}/'; "
                    f"cp -a '{linux_capsule}/manifest.json' '{_wsl_path(destination)}/manifest.json'"
                ),
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise R15BIsolationError(
                f"physical R15B extraction failed ({completed.returncode}): {completed.stderr[-4000:]}"
            )
        result_path = destination / "result.json"
        if not result_path.is_file():
            raise R15BIsolationError("physical R15B extraction result missing")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        launcher = {
            "format": "abi-r15b-isolated-extraction-launcher/1",
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
