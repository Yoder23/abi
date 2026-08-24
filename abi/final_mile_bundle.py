"""Build the content-addressed external clean-room reproduction archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .final_mile import FinalMileError, sha256_file

ABI_RUNTIME_COMMIT = "4b0004bd7b71654a6014d96c634d4418b957d861"
LAYERCAKE_RUNTIME_COMMIT = "662c5a9b7264a1a5478c9dfb656f35c450e2504f"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalMileError(f"expected JSON object: {path}")
    return value


def _git_blob(repository: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            "show",
            f"{commit}:{relative}",
        ],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def _git_paths(repository: Path, commit: str, *prefixes: str) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            *prefixes,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def _custody_key(path: Path) -> Ed25519PrivateKey:
    value = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(value, Ed25519PrivateKey):
        raise FinalMileError("bundle custody key is not Ed25519")
    return value


def _runtime_environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "abi-capability-compiler",
        "cryptography",
        "numpy",
        "psutil",
        "safetensors",
        "torch",
        "transformers",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "format": "abi-final-mile-runtime-environment/1",
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "installation_boundary": (
            "The ABI wheel and exact LayerCake export are supplied. Hardware-specific "
            "PyTorch provisioning remains the independent operator's responsibility."
        ),
    }


def _instructions(wheel_name: str) -> str:
    return f"""# ABI external clean-room reproduction

This archive contains exact runtime exports at ABI `{ABI_RUNTIME_COMMIT}` and
LayerCake `{LAYERCAKE_RUNTIME_COMMIT}`, the three content-addressed payloads
that were not present in those Git trees, the sealed failed-candidate release
family, and the final-mile operator CLI wheel.

Use a genuinely independent operator and different CPU/CUDA hardware. Do not
run CPU or CUDA on the development laptop; the CLI rejects its fingerprint.
Inspect `operator/runtime-environment.json` first. Provision the appropriate
PyTorch build for the independent host, then install the supplied packages.

```text
python -m pip install -e ../layercake_release
python -m pip install operator/{wheel_name} psutil
cd abi_release
abi-reproduce verify
abi-reproduce cpu
abi-reproduce cuda
abi-reproduce quality
abi-reproduce portability
abi-reproduce report
```

Complete `external-reproduction/operator-attestation.json` before CPU or CUDA.
Preserve the first returned outputs even when they fail. The current
portability command is expected to report `HOST_INDEPENDENCE_FAILED`; external
hardware cannot convert that result into a pass.
"""


def build_bundle(
    root: Path,
    *,
    wheel: Path,
    release_dir: Path,
    custody_key: Path,
    output: Path,
) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise FinalMileError("immutable clean-room archive or receipt already exists")
    layercake = (root.parent / "layercake_release").resolve()
    readiness = _object(
        root / "results/abi_capability_compiler_phase8/readiness_v1073/manifest.json"
    )
    with tempfile.TemporaryDirectory(prefix="abi-final-mile-") as temporary:
        staging = Path(temporary) / "abi-final-mile-cleanroom"
        abi_target = staging / "abi_release"
        layercake_target = staging / "layercake_release"
        abi_target.mkdir(parents=True)
        layercake_target.mkdir(parents=True)
        for relative, binding in readiness["files"].items():
            if relative.startswith("../layercake_release/"):
                repository = layercake
                commit = LAYERCAKE_RUNTIME_COMMIT
                repository_relative = relative.removeprefix("../layercake_release/")
                target = layercake_target / repository_relative
            else:
                repository = root
                commit = ABI_RUNTIME_COMMIT
                repository_relative = relative
                target = abi_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if binding.get("tracked") is False:
                source = (root / relative).resolve()
                if sha256_file(source) != binding["sha256"]:
                    raise FinalMileError(f"untracked release payload changed: {relative}")
                target.write_bytes(source.read_bytes())
            else:
                target.write_bytes(_git_blob(repository, commit, repository_relative))
            if sha256_file(target) != binding["sha256"]:
                raise FinalMileError(f"runtime commit export changed: {relative}")

        layercake_sources = _git_paths(
            layercake,
            LAYERCAKE_RUNTIME_COMMIT,
            "layercake",
            "layercake_extensions",
        )
        layercake_sources.extend(["pyproject.toml"])
        for relative in sorted(set(layercake_sources)):
            if not (relative.endswith(".py") or relative == "pyproject.toml"):
                continue
            target = layercake_target / relative
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_git_blob(layercake, LAYERCAKE_RUNTIME_COMMIT, relative))

        release_target = abi_target / "results/abi_final_mile/abi-release"
        for source in sorted(release_dir.resolve().rglob("*")):
            if source.is_file():
                target = release_target / source.relative_to(release_dir.resolve())
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
        operator = staging / "operator"
        operator.mkdir(parents=True)
        (operator / wheel.name).write_bytes(wheel.resolve().read_bytes())
        (operator / "operator-attestation-template.json").write_bytes(
            (root / "PHASE8_EXTERNAL_OPERATOR_ATTESTATION_TEMPLATE_V1.json").read_bytes()
        )
        (operator / "runtime-environment.json").write_bytes(
            json.dumps(_runtime_environment(), indent=2, sort_keys=True).encode() + b"\n"
        )
        external = abi_target / "external-reproduction"
        external.mkdir(parents=True, exist_ok=True)
        (external / "operator-attestation.json").write_bytes(
            (operator / "operator-attestation-template.json").read_bytes()
        )
        (staging / "README.md").write_text(_instructions(wheel.name), encoding="utf-8", newline="\n")

        inventory = {}
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name not in {"bundle-manifest.json", "bundle-signature.json"}:
                relative = path.relative_to(staging).as_posix()
                inventory[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "format": "abi-final-mile-cleanroom-bundle/1",
            "status": "SEALED_EXTERNAL_HANDOFF_HOST_INDEPENDENCE_FAILED",
            "abi_runtime_commit": ABI_RUNTIME_COMMIT,
            "layercake_runtime_commit": LAYERCAKE_RUNTIME_COMMIT,
            "development_hardware_fingerprint_sha256": readiness["development_hardware"][
                "fingerprint_sha256"
            ],
            "files": inventory,
            "file_count": len(inventory),
            "total_bytes": sum(row["bytes"] for row in inventory.values()),
            "external_human_gate_included": False,
            "release_certified": False,
            "host_independence_status": "HOST_INDEPENDENCE_FAILED",
        }
        manifest_path = staging / "bundle-manifest.json"
        manifest_path.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
        private = _custody_key(custody_key.resolve())
        public = private.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        )
        manifest_bytes = manifest_path.read_bytes()
        signature = {
            "format": "abi-final-mile-cleanroom-bundle-signature/1",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "signature_ed25519_hex": private.sign(manifest_bytes).hex(),
            "public_key_pem": public.decode("ascii"),
        }
        (staging / "bundle-signature.json").write_bytes(
            json.dumps(signature, indent=2, sort_keys=True).encode() + b"\n"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())

    receipt = {
        "format": "abi-final-mile-cleanroom-archive-receipt/1",
        "status": "PASS_CONTENT_ADDRESSED_CLEANROOM_ARCHIVE_BUILT",
        "archive": output.name,
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "manifest_sha256": signature["manifest_sha256"],
        "files": manifest["file_count"],
        "uncompressed_bytes": manifest["total_bytes"],
        "abi_runtime_commit": ABI_RUNTIME_COMMIT,
        "layercake_runtime_commit": LAYERCAKE_RUNTIME_COMMIT,
        "host_independence_status": "HOST_INDEPENDENCE_FAILED",
        "release_certified": False,
    }
    receipt_path = output.with_suffix(output.suffix + ".json")
    receipt_path.write_bytes(json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
    return receipt


def verify_bundle(path: Path) -> dict[str, Any]:
    """Stream-verify archive paths, inventory hashes, and outer signature."""
    path = path.resolve()
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise FinalMileError("clean-room archive contains duplicate paths")
        if any(
            name.startswith("/") or ".." in Path(name).parts or "\\" in name
            for name in names
        ):
            raise FinalMileError("clean-room archive contains an unsafe path")
        manifest_bytes = archive.read("bundle-manifest.json")
        manifest = json.loads(manifest_bytes)
        signature = json.loads(archive.read("bundle-signature.json"))
        public = serialization.load_pem_public_key(signature["public_key_pem"].encode())
        if not isinstance(public, Ed25519PublicKey):
            raise FinalMileError("bundle signature key is not Ed25519")
        if hashlib.sha256(manifest_bytes).hexdigest() != signature.get("manifest_sha256"):
            raise FinalMileError("bundle manifest digest changed")
        public.verify(bytes.fromhex(signature["signature_ed25519_hex"]), manifest_bytes)
        declared = manifest.get("files")
        actual_names = set(names) - {"bundle-manifest.json", "bundle-signature.json"}
        if not isinstance(declared, dict) or set(declared) != actual_names:
            raise FinalMileError("bundle file inventory changed")
        for name, binding in declared.items():
            digest = hashlib.sha256()
            size = 0
            with archive.open(name, "r") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if size != binding.get("bytes") or digest.hexdigest() != binding.get("sha256"):
                raise FinalMileError(f"bundle member changed: {name}")
    return {
        "format": "abi-final-mile-cleanroom-archive-verification/1",
        "status": "PASS_CLEANROOM_ARCHIVE_SIGNATURE_PATHS_AND_HASHES",
        "archive_sha256": sha256_file(path),
        "archive_bytes": path.stat().st_size,
        "files": len(declared),
        "uncompressed_bytes": sum(row["bytes"] for row in declared.values()),
        "host_independence_status": manifest["host_independence_status"],
        "release_certified": manifest["release_certified"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--custody-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = build_bundle(
        root,
        wheel=(root / args.wheel).resolve(),
        release_dir=(root / args.release_dir).resolve(),
        custody_key=(root / args.custody_key).resolve(),
        output=(root / args.output).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
