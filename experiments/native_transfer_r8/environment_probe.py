"""Record immutable R8 hardware, model, and isolation-runtime preflight evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

from .capability_generator import canonical_json_bytes
from .native_host import SPECS, sha256_file, snapshot_inventory


class EnvironmentProbeError(RuntimeError):
    """Raised when an immutable preflight output would be overwritten."""


def _command(executable: str, arguments: list[str]) -> dict[str, Any]:
    path = shutil.which(executable)
    if path is None:
        return {"available": False, "executable": None, "returncode": None, "stdout": ""}
    completed = subprocess.run(
        [path, *arguments], check=False, capture_output=True, text=True, timeout=30
    )
    return {
        "available": True,
        "executable": path,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def probe(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise EnvironmentProbeError(f"immutable environment probe exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    models = {}
    for key in ("source", *sorted(config["models"]["recipients"])):
        spec = SPECS[key]
        try:
            snapshot = Path(
                snapshot_download(
                    spec.model_id, revision=spec.revision, local_files_only=True
                )
            ).resolve()
            models[key] = {
                "available": True,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "snapshot": str(snapshot),
                "inventory": snapshot_inventory(snapshot),
            }
        except Exception as exc:  # fail-closed evidence records the concrete local exception
            models[key] = {
                "available": False,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    packages = {}
    for name in ("torch", "transformers", "huggingface-hub", "safetensors", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    cuda = {
        "available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    isolation = {
        "docker": _command("docker", ["version", "--format", "{{json .}}"]),
        "podman": _command("podman", ["version", "--format", "json"]),
    }
    value = {
        "format": "abi-native-transfer-r8-environment-preflight/1",
        "created_unix_time_ns": time.time_ns(),
        "config_sha256": sha256_file(config_path),
        "python": sys.version,
        "platform": platform.platform(),
        "node": platform.node(),
        "packages": packages,
        "cuda": cuda,
        "models": models,
        "isolation_runtimes": isolation,
        "physical_isolation_available": any(
            row["available"] and row["returncode"] == 0 for row in isolation.values()
        ),
    }
    value["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        value = probe(Path(args.config).resolve(), Path(args.output).resolve())
    except EnvironmentProbeError as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}, indent=2))
        return 2
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
