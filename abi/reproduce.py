"""External-only command surface for ABI clean-room reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .capability_compiler_phase7_direct_artifact_runtime import load_protocol
from .capability_compiler_phase7_verify import _rows, verify_device_document
from .capability_compiler_phase8_release_readiness import hardware_document
from .final_mile import FinalMileError, sha256_file

DEVELOPMENT_HARDWARE_SHA256 = "2deae6043290bb50a87b17e799b7772af1df02356f3f954c6c755bb85782f02a"
PRODUCT_PROTOCOL = "ABI_CAPABILITY_COMPILER_PHASE7_ALLOCATION_BOUNDED_VERIFY_PROTOCOL_V1061.json"
RUN_ROOT = Path("external-reproduction/raw")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FinalMileError(f"expected JSON object: {path}")
    return value


def verify_release(release_dir: Path) -> dict[str, Any]:
    release_dir = release_dir.resolve()
    manifest_path = release_dir / "release-manifest.json"
    signature_path = release_dir / "release-signature.json"
    manifest, signature = _object(manifest_path), _object(signature_path)
    manifest_bytes = manifest_path.read_bytes()
    if (
        manifest.get("format") != "abi-final-mile-release-manifest/1"
        or signature.get("format") != "abi-final-mile-release-signature/1"
        or hashlib.sha256(manifest_bytes).hexdigest() != signature.get("manifest_sha256")
    ):
        raise FinalMileError("release manifest or signature binding changed")
    public = serialization.load_pem_public_key(signature["public_key_pem"].encode())
    if not isinstance(public, Ed25519PublicKey):
        raise FinalMileError("release signature key is not Ed25519")
    public.verify(bytes.fromhex(signature["signature_ed25519_hex"]), manifest_bytes)
    files = manifest.get("files")
    if not isinstance(files, dict) or len(files) != manifest.get("file_count"):
        raise FinalMileError("release inventory depth changed")
    for relative, binding in files.items():
        path = (release_dir / relative).resolve()
        if (
            not path.is_relative_to(release_dir)
            or not path.is_file()
            or path.stat().st_size != binding.get("bytes")
            or sha256_file(path) != binding.get("sha256")
        ):
            raise FinalMileError(f"release inventory changed: {relative}")
    return {
        "format": "abi-reproduce-verification/1",
        "status": "PASS_RELEASE_BYTES_AND_OUTER_SIGNATURE",
        "manifest_sha256": sha256_file(manifest_path),
        "files": len(files),
        "bytes": sum(row["bytes"] for row in files.values()),
        "release_certified": manifest.get("release_certified") is True,
        "claim_boundary": (
            "Byte verification does not override a failed or incomplete release certificate."
        ),
    }


def _validate_attestation(path: Path) -> dict[str, Any]:
    value = _object(path)
    required_true = (
        "independent_of_abi_development",
        "independent_hardware_owned_or_controlled_by_operator",
        "clean_abi_commit_verified",
        "clean_layercake_commit_verified",
        "release_manifest_verified_before_execution",
        "artifact_hashes_verified_before_execution",
    )
    if (
        value.get("format")
        != "abi-capability-compiler-phase8-external-operator-attestation/1"
        or not isinstance(value.get("operator_id"), str)
        or not value["operator_id"].strip()
        or any(value.get(field) is not True for field in required_true)
    ):
        raise FinalMileError("external operator attestation is incomplete")
    return value


def external_preflight(*, attestation: Path, require_cuda: bool) -> dict[str, Any]:
    operator = _validate_attestation(attestation)
    hardware = hardware_document()
    different = hardware["fingerprint_sha256"] != DEVELOPMENT_HARDWARE_SHA256
    if not different:
        raise FinalMileError("development hardware cannot be used for external reproduction")
    if require_cuda and hardware.get("cuda") is None:
        raise FinalMileError("CUDA reproduction requires a CUDA-capable external GPU")
    return {
        "operator_id": operator["operator_id"],
        "hardware": hardware,
        "different_from_development_hardware": different,
    }


def _run(command: list[str], *, root: Path) -> None:
    subprocess.run(command, cwd=root, check=True)


def run_device(root: Path, *, device: str, attestation: Path) -> dict[str, Any]:
    preflight = external_preflight(attestation=attestation, require_cuda=device == "cuda")
    output = root / RUN_ROOT / device
    if output.exists():
        raise FinalMileError(f"immutable first external {device} result already exists")
    command = [
        sys.executable,
        "-m",
        "abi.capability_compiler_phase7_direct_artifact_runtime",
        "--protocol",
        PRODUCT_PROTOCOL,
        "--device",
        device,
        "--output-dir",
        output.relative_to(root).as_posix(),
    ]
    _run(command, root=root)
    result = _object(output / "result.json")
    return {
        "format": "abi-reproduce-device-result/1",
        "status": f"PASS_EXTERNAL_{device.upper()}_COLLECTION"
        if result.get("status") == "PASS_PHASE7_INTEGRATED_RUNTIME"
        else f"FAIL_EXTERNAL_{device.upper()}_COLLECTION",
        "device": device,
        "preflight": preflight,
        "result_sha256": sha256_file(output / "result.json"),
        "observations_sha256": sha256_file(output / "observations.jsonl"),
        "raw_result": result,
    }


def _identity(rows: list[Mapping[str, Any]]) -> list[tuple[str, str, tuple[int, ...]]]:
    return sorted(
        (
            str(row.get("mode")),
            str(row.get("probe_id")),
            tuple(int(value) for value in row.get("output_token_ids", [])),
        )
        for row in rows
        if row.get("mode") in {"core_runtime", "domain_runtime"}
    )


def verify_quality(root: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, root / PRODUCT_PROTOCOL)
    devices = {}
    identities = {}
    for device in ("cpu", "cuda"):
        directory = root / RUN_ROOT / device
        result = _object(directory / "result.json")
        observations = _rows(directory / "observations.jsonl")
        gates = verify_device_document(
            root=root,
            protocol=protocol,
            protocol_sha=protocol_sha,
            device=device,
            result=result,
            observations=observations,
        )
        devices[device] = {"gates": gates, "passed": all(gates.values())}
        identities[device] = _identity(observations)
    cross = identities["cpu"] == identities["cuda"]
    passed = all(row["passed"] for row in devices.values()) and cross and len(identities["cpu"]) == 240
    return {
        "format": "abi-reproduce-quality-result/1",
        "status": "PASS_EXTERNAL_QUALITY_AND_RUNTIME_RECOMPUTATION"
        if passed
        else "FAIL_EXTERNAL_QUALITY_AND_RUNTIME_RECOMPUTATION",
        "devices": devices,
        "cross_device_runtime_identities": len(identities["cpu"]) if cross else 0,
        "teacher_loaded_by_verifier": False,
        "training_performed_by_verifier": False,
    }


def verify_portability(release_dir: Path) -> dict[str, Any]:
    matrix = _object(release_dir / "compatibility-matrix.json")
    if matrix.get("status") == "PASS_CROSS_HOST_PORTABILITY":
        status = "PASS_EXTERNAL_PORTABILITY_INPUT"
    else:
        status = "HOST_INDEPENDENCE_FAILED"
    return {
        "format": "abi-reproduce-portability-result/1",
        "status": status,
        "source_status": matrix.get("status"),
        "receivers_passing": matrix.get("receivers_passing"),
        "receivers_required": matrix.get("receivers_required"),
        "claim_boundary": "External hardware cannot repair an artifact/receiver ABI incompatibility.",
    }


def build_report(root: Path, release_dir: Path, attestation: Path) -> dict[str, Any]:
    verification = verify_release(release_dir)
    operator = _validate_attestation(attestation)
    portability = verify_portability(release_dir)
    quality_path = root / RUN_ROOT / "quality.json"
    quality = _object(quality_path) if quality_path.exists() else {"status": "NOT_RUN"}
    passed = (
        verification["status"].startswith("PASS")
        and quality.get("status") == "PASS_EXTERNAL_QUALITY_AND_RUNTIME_RECOMPUTATION"
        and portability["status"] == "PASS_EXTERNAL_PORTABILITY_INPUT"
    )
    return {
        "format": "abi-reproduce-report/1",
        "status": "PASS_EXTERNAL_REPRODUCTION" if passed else "FAIL_EXTERNAL_REPRODUCTION",
        "operator_id": operator["operator_id"],
        "release": verification,
        "quality": quality,
        "portability": portability,
        "phase8_certified": False,
        "reason_phase8_not_certified": (
            "Custodian hostile verification, human ratings, and final certificate remain separate."
        ),
    }


def _write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FinalMileError(f"immutable external result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="abi-reproduce", description=__doc__)
    parser.add_argument("command", choices=("verify", "cpu", "cuda", "quality", "portability", "report"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--release-dir", default="results/abi_final_mile/abi-release")
    parser.add_argument("--attestation", default="external-reproduction/operator-attestation.json")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    release = (root / args.release_dir).resolve()
    attestation = (root / args.attestation).resolve()
    if args.command == "verify":
        result = verify_release(release)
    elif args.command in {"cpu", "cuda"}:
        result = run_device(root, device=args.command, attestation=attestation)
    elif args.command == "quality":
        result = verify_quality(root)
        _write_once(root / RUN_ROOT / "quality.json", result)
    elif args.command == "portability":
        result = verify_portability(release)
        _write_once(root / RUN_ROOT / "portability.json", result)
    else:
        result = build_report(root, release, attestation)
        _write_once(root / RUN_ROOT / "report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
