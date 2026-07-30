"""Teacher-free LayerCake product host for one English core and signed cakes."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from typing import Any, Mapping
import zipfile

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .layercake_host_runtime import NativeHostRuntime, generate_native_host


PRODUCT_HOST_FORMAT = "abi-layercake-product-host/1"
DIRECT_ABI_VERSION = "lc-direct-neural-decoder/1"
DIRECT_ABI_SHA256 = (
    "de765899700aefe22bfe6c9d00ed5b0c1f87a7ef864cf7211aa8aa4491a0742a"
)
MAX_PACKAGE_BYTES = 512 * 1024 * 1024
CONTENT_CONTEXT = b"LAYERCAKE-CAKE-CONTENT-V1\0"
SIGNING_CONTEXT = b"LAYERCAKE-CAKE-SIGNATURE-V1\0"


class ProductHostError(RuntimeError):
    """Raised when product-host identity or execution fails closed."""


@dataclass(frozen=True)
class ProductHostResult:
    engine: str
    output: str
    output_sha256: str
    cake_id: str | None
    evidence: dict[str, Any]


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_key(raw: bytes) -> tuple[Ed25519PublicKey, str]:
    try:
        key = serialization.load_pem_public_key(raw)
    except ValueError as exc:
        raise ProductHostError("publisher key is not valid PEM") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise ProductHostError("publisher key is not Ed25519")
    encoded = key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return key, hashlib.sha256(encoded).hexdigest()[:32]


def _content_hash(manifest: Mapping[str, Any], payload: bytes) -> str:
    unsigned = dict(manifest)
    unsigned["package_hash"] = ""
    manifest_bytes = _canonical_json(unsigned)
    digest = hashlib.sha256()
    digest.update(CONTENT_CONTEXT)
    for name, data in (
        ("manifest.json", manifest_bytes),
        ("tensors.safetensors", payload),
    ):
        encoded = name.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def verify_domain_package(
    package_path: str | Path,
    public_key_path: str | Path,
) -> dict[str, Any]:
    """Verify a signed domain package without importing its PyTorch runtime."""

    package_path = Path(package_path).resolve()
    public_key_path = Path(public_key_path).resolve()
    if package_path.suffix != ".cake" or not package_path.is_file():
        raise ProductHostError("domain package must be an existing .cake file")
    raw = package_path.read_bytes()
    if len(raw) > MAX_PACKAGE_BYTES:
        raise ProductHostError("domain package exceeds the size limit")
    try:
        with zipfile.ZipFile(package_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            folded = [name.casefold() for name in names]
            if names != [
                "manifest.json",
                "tensors.safetensors",
                "signature.json",
            ]:
                raise ProductHostError("domain package members are not canonical")
            if len(names) != len(set(names)) or len(folded) != len(set(folded)):
                raise ProductHostError("domain package has ambiguous members")
            for info in infos:
                member = PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or len(member.parts) != 1
                    or info.is_dir()
                    or info.flag_bits & 0x1
                ):
                    raise ProductHostError("domain package has an unsafe member")
            manifest_raw = archive.read("manifest.json")
            payload = archive.read("tensors.safetensors")
            signature_raw = archive.read("signature.json")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ProductHostError("domain package is not a valid archive") from exc
    try:
        manifest = json.loads(manifest_raw)
        signature = json.loads(signature_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductHostError("domain package metadata is invalid JSON") from exc
    if not isinstance(manifest, dict) or not isinstance(signature, dict):
        raise ProductHostError("domain package metadata must be objects")
    domains = manifest.get("domains")
    if (
        manifest.get("schema_version") != "1"
        or manifest.get("abi_version") != DIRECT_ABI_VERSION
        or manifest.get("abi_hash") != DIRECT_ABI_SHA256
        or manifest.get("cake_type") != "portable_decoder"
        or manifest.get("input_contract", {}).get("mode")
        != "direct_selected_portable_decoder"
        or not isinstance(domains, list)
        or len(domains) != 1
        or not isinstance(domains[0], str)
        or not domains[0]
    ):
        raise ProductHostError("domain package manifest is incompatible")
    payload_hash = hashlib.sha256(payload).hexdigest()
    if payload_hash != manifest.get("tensor_payload_hash"):
        raise ProductHostError("domain package tensor payload changed")
    package_hash = _content_hash(manifest, payload)
    if package_hash != manifest.get("package_hash"):
        raise ProductHostError("domain package content hash changed")
    if set(signature) != {
        "algorithm",
        "key_id",
        "signed_hash",
        "signature",
    } or signature.get("algorithm") != "ed25519":
        raise ProductHostError("domain package signature schema is invalid")
    public_raw = public_key_path.read_bytes()
    public, key_id = _public_key(public_raw)
    if (
        signature.get("key_id") != key_id
        or manifest.get("signature", {}).get("key_id") != key_id
        or signature.get("signed_hash") != package_hash
    ):
        raise ProductHostError("domain package publisher identity differs")
    try:
        signature_bytes = base64.b64decode(
            signature["signature"], validate=True
        )
        public.verify(
            signature_bytes,
            SIGNING_CONTEXT + package_hash.encode("ascii"),
        )
    except (InvalidSignature, ValueError, TypeError) as exc:
        raise ProductHostError("domain package signature is invalid") from exc
    return {
        "cake_id": str(manifest["cake_id"]),
        "domain": domains[0],
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "archive_bytes": len(raw),
        "package_hash": package_hash,
        "tensor_payload_hash": payload_hash,
        "key_id": key_id,
        "public_key_sha256": hashlib.sha256(public_raw).hexdigest(),
        "package_path": str(package_path),
        "public_key_path": str(public_key_path),
        "signed": True,
        "teacher_present": False,
        "source_transformer_blocks": 0,
    }


class LayerCakeProductHost:
    """One native English core plus explicitly selected, lazy domain workers."""

    def __init__(
        self,
        *,
        english_artifact: str | Path,
        layercake_root: str | Path,
        registry_root: str | Path,
        threads: int = 4,
    ) -> None:
        self.english_artifact = Path(english_artifact).resolve()
        self.layercake_root = Path(layercake_root).resolve()
        self.registry_root = Path(registry_root).resolve()
        self.registry_root.mkdir(parents=True, exist_ok=True)
        self.english_runtime = NativeHostRuntime(
            self.english_artifact, threads=threads
        )
        self._packages: dict[str, dict[str, Any]] = {}
        self._workers: dict[str, subprocess.Popen[str]] = {}
        self._english_calls = 0

    def install(
        self,
        package_path: str | Path,
        public_key_path: str | Path,
    ) -> dict[str, Any]:
        verified = verify_domain_package(package_path, public_key_path)
        cake_id = verified["cake_id"]
        destination = self.registry_root / (
            f"{verified['archive_sha256']}.cake"
        )
        key_destination = self.registry_root / f"{verified['key_id']}.pub"
        if not destination.exists():
            shutil.copyfile(verified["package_path"], destination)
        if not key_destination.exists():
            shutil.copyfile(verified["public_key_path"], key_destination)
        if (
            _sha256_file(destination) != verified["archive_sha256"]
            or _sha256_file(key_destination)
            != verified["public_key_sha256"]
        ):
            raise ProductHostError("content-addressed registry write changed bytes")
        registered = {
            **verified,
            "package_path": str(destination),
            "public_key_path": str(key_destination),
        }
        self._packages[cake_id] = registered
        return {"status": "INSTALLED", **registered}

    def installed(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(self._packages[key]) for key in sorted(self._packages))

    def _worker(self, device: str) -> subprocess.Popen[str]:
        current = self._workers.get(device)
        if current is not None and current.poll() is None:
            return current
        creationflags = (
            subprocess.CREATE_NO_WINDOW
            if os.name == "nt"
            else 0
        )
        worker = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "abi.layercake_domain_worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags,
        )
        assert worker.stdin is not None
        assert worker.stdout is not None
        configuration = {
            "layercake_root": str(self.layercake_root),
            "registry_root": str(self.registry_root / f"worker-{device}"),
            "device": device,
            "packages": list(self.installed()),
        }
        worker.stdin.write(json.dumps(configuration) + "\n")
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            error = worker.stderr.read() if worker.stderr is not None else ""
            worker.kill()
            raise ProductHostError(f"domain worker failed to start: {error}")
        ready = json.loads(line)
        if ready.get("status") != "READY":
            worker.kill()
            raise ProductHostError(f"domain worker rejected setup: {ready}")
        self._workers[device] = worker
        return worker

    def generate(
        self,
        prompt: str,
        *,
        cake_id: str | None = None,
        max_new_tokens: int = 160,
        domain_device: str = "cpu",
    ) -> ProductHostResult:
        if cake_id is None:
            generated = generate_native_host(
                self.english_runtime,
                prompt,
                max_new_tokens=max_new_tokens,
            )
            self._english_calls += 1
            return ProductHostResult(
                engine="native_english",
                output=generated["output"],
                output_sha256=generated["output_sha256"],
                cake_id=None,
                evidence=generated,
            )
        if cake_id not in self._packages:
            raise ProductHostError(f"domain cake is not installed: {cake_id}")
        worker = self._worker(domain_device)
        assert worker.stdin is not None
        assert worker.stdout is not None
        worker.stdin.write(
            json.dumps(
                {
                    "command": "generate",
                    "cake_id": cake_id,
                    "prompt": prompt,
                    "maximum_actions": max_new_tokens,
                }
            )
            + "\n"
        )
        worker.stdin.flush()
        line = worker.stdout.readline()
        if not line:
            error = worker.stderr.read() if worker.stderr is not None else ""
            raise ProductHostError(f"domain worker stopped unexpectedly: {error}")
        response = json.loads(line)
        if response.get("status") != "PASS":
            raise ProductHostError(
                f"domain worker rejected generation: {response}"
            )
        output = base64.b64decode(response["output_base64"]).decode("utf-8")
        return ProductHostResult(
            engine=f"domain_worker_{domain_device}",
            output=output,
            output_sha256=response["output_sha256"],
            cake_id=cake_id,
            evidence=response,
        )

    def telemetry(self) -> dict[str, Any]:
        return {
            "format": PRODUCT_HOST_FORMAT,
            "english_calls": self._english_calls,
            "installed_cakes": sorted(self._packages),
            "active_domain_worker_devices": sorted(
                device
                for device, worker in self._workers.items()
                if worker.poll() is None
            ),
            "maximum_active_domain_cakes_per_request": 1,
        }

    def close(self) -> None:
        for worker in self._workers.values():
            if worker.poll() is not None:
                continue
            assert worker.stdin is not None
            worker.stdin.write(json.dumps({"command": "close"}) + "\n")
            worker.stdin.flush()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
        self._workers.clear()

    def __enter__(self) -> "LayerCakeProductHost":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
