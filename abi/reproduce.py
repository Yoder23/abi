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
    if value.get("format") == "abi-final-external-operator-attestation/1":
        required_true = (
            "independent_of_abi_development",
            "independent_hardware_owned_or_controlled_by_operator",
            "different_from_development_hardware",
            "clean_archive_hash_verified_before_execution",
            "frozen_commit_and_tag_verified",
            "artifact_hashes_verified_before_execution",
            "capability_files_not_exposed_during_host_certification",
            "teacher_or_source_model_not_executed",
            "raw_failures_preserved",
        )
        if (
            not isinstance(value.get("operator_id"), str)
            or not value["operator_id"].strip()
            or any(value.get(field) is not True for field in required_true)
        ):
            raise FinalMileError("external operator attestation is incomplete")
        return value
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


def verify_final_validation(root: Path) -> dict[str, Any]:
    from abi_v2.final_validation import (
        CAPABILITY_PATHS,
        evidence_hash,
        read_json,
        recompute_headlines,
    )
    from abi_v2.final_validation import (
        sha256_file as final_sha256,
    )

    candidate_path = root / "results/abi_final_validation/frozen_release_candidate.json"
    candidate = read_json(candidate_path)
    failures: list[str] = []
    if candidate.get("evidence_sha256") != evidence_hash(candidate):
        failures.append("frozen_candidate_self_hash")
    for name, row in candidate["evaluator_and_data_bindings"].items():
        path = root / row["path"]
        if not path.is_file() or final_sha256(path) != row["sha256"]:
            failures.append(f"binding:{name}")
    for name, relative in CAPABILITY_PATHS.items():
        if final_sha256(root / relative) != candidate["capability_artifacts"][name]["sha256"]:
            failures.append(f"capability:{name}")
    for host, row in candidate["host_adapters"].items():
        if final_sha256(root / row["path"]) != row["sha256"]:
            failures.append(f"adapter:{host}")
    recomputed = recompute_headlines(root)
    locked = read_json(root / "results/abi_final_validation/headline_recomputation.json")
    if recomputed["aggregate"] != locked["aggregate"]:
        failures.append("headline_recomputation")
    return {
        "format": "abi-reproduce-final-byte-verification/1",
        "status": "PASS_FINAL_VALIDATION_BYTES_AND_RAW_RECOMPUTATION"
        if not failures
        else "FAIL_FINAL_VALIDATION_BYTES",
        "failures": failures,
        "frozen_commit": candidate["technical_proof_commit"],
        "frozen_tag": candidate["technical_proof_tag"],
        "headline_aggregate": recomputed["aggregate"],
    }


def _final_output(root: Path) -> Path:
    return root / "external_reproduction/raw/final"


def certify_final_hosts(
    root: Path, *, attestation: Path, qwen_snapshot: Path, pythia_snapshot: Path, device: str
) -> dict[str, Any]:
    from abi_v2.final_validation import read_json
    from abi_v2.final_validation import sha256_file as final_sha256
    from abi_v2.host_certification import certify_host

    preflight = external_preflight(attestation=attestation, require_cuda=device == "cuda")
    target = _final_output(root) / "host_certification"
    if target.exists():
        raise FinalMileError("immutable external host-certification result already exists")
    candidate = read_json(root / "results/abi_final_validation/frozen_release_candidate.json")
    results = {}
    for host, snapshot, selected_device in (
        ("layercake", None, "cpu"),
        ("qwen2", qwen_snapshot, device),
        ("pythia", pythia_snapshot, device),
    ):
        result = certify_host(
            root,
            host_key=host,
            output_dir=target / host,
            snapshot=snapshot,
            device=selected_device,
        )
        adapter = target / host / "adapter.json"
        expected = candidate["host_adapters"][host]["sha256"]
        results[host] = {
            "status": result["status"],
            "adapter_sha256": final_sha256(adapter),
            "expected_adapter_sha256": expected,
            "adapter_exact": final_sha256(adapter) == expected,
        }
    passed = all(row["status"].startswith("PASS") and row["adapter_exact"] for row in results.values())
    return {
        "format": "abi-reproduce-final-host-certification/1",
        "status": "PASS_EXTERNAL_CAPABILITY_BLIND_HOST_CERTIFICATIONS"
        if passed
        else "FAIL_EXTERNAL_HOST_CERTIFICATION",
        "preflight": preflight,
        "hosts": results,
    }


def run_final_matrix(
    root: Path, *, attestation: Path, qwen_snapshot: Path, pythia_snapshot: Path, device: str
) -> dict[str, Any]:
    from abi_v2.capability_matrix import run
    from abi_v2.final_validation import HOSTS, read_json
    from abi_v2.final_validation import sha256_file as final_sha256

    preflight = external_preflight(attestation=attestation, require_cuda=device == "cuda")
    target = _final_output(root) / "matrix"
    if target.exists():
        raise FinalMileError("immutable external capability-matrix result already exists")
    candidate = read_json(root / "results/abi_final_validation/frozen_release_candidate.json")
    certification_root = _final_output(root) / "host_certification"
    certification_receipt = _object(_final_output(root) / "certify-hosts.json")
    if not certification_receipt.get("status", "").startswith("PASS"):
        raise FinalMileError("fresh capability-blind host certification did not pass")
    for host in HOSTS:
        adapter = certification_root / host / "adapter.json"
        expected = candidate["host_adapters"][host]["sha256"]
        if not adapter.is_file() or final_sha256(adapter) != expected:
            raise FinalMileError(f"fresh frozen adapter missing or changed: {host}")
    results = {}
    for host, snapshot in (
        ("layercake", None),
        ("qwen2", qwen_snapshot),
        ("pythia", pythia_snapshot),
    ):
        result = run(
            root,
            protocol_path=root / "abi_v2/matrix_protocol_amendment3.json",
            host_key=host,
            output_dir=target / host,
            snapshot=snapshot,
            device=device,
        )
        results[host] = {"status": result["status"], "gates": result["gates"]}
    passed = all(row["status"].startswith("PASS") and all(row["gates"].values()) for row in results.values())
    return {
        "format": "abi-reproduce-final-capability-matrix/1",
        "status": "PASS_EXTERNAL_3_HOST_4_CAPABILITY_MATRIX"
        if passed
        else "FAIL_EXTERNAL_CAPABILITY_MATRIX",
        "preflight": preflight,
        "hosts": results,
    }


def final_causality(root: Path) -> dict[str, Any]:
    from abi_v2.final_validation import CAPABILITIES, HOSTS, read_json, read_jsonl, sha256_bytes

    matrix_root = _final_output(root) / "matrix"
    outputs = {}
    removal = 0
    adapter = 0
    for host in HOSTS:
        rows = read_jsonl(matrix_root / host / "observations.jsonl")
        result = read_json(matrix_root / host / "result.json")
        outputs[host] = {
            (str(row["capability"]), str(row["probe_id"])): str(row["output"])
            for row in rows
            if sha256_bytes(str(row["output"]).encode("utf-8")) == row["output_sha256"]
        }
        removal += sum(
            bool(row["absent_execution_rejected"]) and bool(row["restored_output_byte_exact"])
            for row in result["causal"]["capability_removal_and_reinstall"].values()
        )
        adapter += bool(result["causal"]["adapter_removal"]["rejected"])
    common = set.intersection(*(set(row) for row in outputs.values()))
    equal = sum(len({outputs[host][key] for host in HOSTS}) == 1 for key in common)
    passed = equal == len(common) and removal == len(HOSTS) * len(CAPABILITIES) and adapter == len(HOSTS)
    return {
        "format": "abi-reproduce-final-causality/1",
        "status": "PASS_EXTERNAL_CAUSALITY_WITH_STANDALONE_RUNTIME_BOUNDARY"
        if passed
        else "FAIL_EXTERNAL_CAUSALITY",
        "neutral_stub_exact_outputs": sum(len(row) for row in outputs.values()),
        "cross_host_equal": equal,
        "cross_host_total": len(common),
        "capability_removal_reinstall": removal,
        "adapter_removal": adapter,
        "causal_boundary": "host hidden state is noncausal; capability plus generic runtime is standalone",
    }


def final_performance(root: Path) -> dict[str, Any]:
    from abi_v2.final_validation import HOSTS, _recompute_performance, read_json

    base = _final_output(root)
    hosts = {}
    for host in HOSTS:
        perf = read_json(base / "host_certification" / host / "performance.json")
        matrix = read_json(base / "matrix" / host / "result.json")
        hosts[host] = {
            "adapter": _recompute_performance(perf),
            "memory": {
                "peak_process_rss_bytes_lower_bound": matrix["performance"]["peak_process_rss_bytes_lower_bound"],
                "peak_cuda_allocated_bytes": matrix["performance"]["peak_cuda_allocated_bytes"],
            },
            "installation_seconds": {
                name: row["seconds"] for name, row in matrix["installation"].items()
            },
            "capability_execution": matrix["performance"]["capability_execution"],
        }
    passed = all(row["adapter"]["passes"] for row in hosts.values())
    return {
        "format": "abi-reproduce-final-performance/1",
        "status": "PASS_EXTERNAL_PERFORMANCE_RECOMPUTATION" if passed else "FAIL_EXTERNAL_PERFORMANCE",
        "hosts": hosts,
    }


def final_hostile(root: Path) -> dict[str, Any]:
    from abi_v2.hostile_final_validation import run

    return run(root)


def final_report(root: Path, attestation: Path) -> dict[str, Any]:
    operator = _validate_attestation(attestation)
    base = _final_output(root)
    names = ("verify", "certify-hosts", "capability-matrix", "causality", "performance", "hostile-audit")
    evidence = {name: _object(base / f"{name}.json") for name in names}
    passed = all(row["status"].startswith("PASS") for row in evidence.values())
    return {
        "format": "abi-reproduce-final-report/1",
        "status": "PASS_EXTERNAL_REPRODUCTION_PENDING_CUSTODIAN_AUDIT"
        if passed
        else "FAIL_EXTERNAL_REPRODUCTION",
        "operator_id": operator["operator_id"],
        "evidence": evidence,
        "independent_gate_closed_automatically": False,
        "next_step": "return raw evidence and attestation to the independent custodian",
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="abi-reproduce", description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "verify",
            "certify-hosts",
            "capability-matrix",
            "causality",
            "performance",
            "hostile-audit",
            "cpu",
            "cuda",
            "quality",
            "portability",
            "report",
        ),
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--release-dir", default="results/abi_final_mile/abi-release")
    parser.add_argument("--attestation", default="external_reproduction/operator-attestation.json")
    parser.add_argument("--qwen-snapshot", default="external_reproduction/models/qwen2")
    parser.add_argument("--pythia-snapshot", default="external_reproduction/models/pythia")
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    release = (root / args.release_dir).resolve()
    attestation = (root / args.attestation).resolve()
    final_mode = (root / "results/abi_final_validation/frozen_release_candidate.json").is_file()
    final_target = _final_output(root)
    if args.command == "verify" and final_mode:
        result = verify_final_validation(root)
        _write_once(final_target / "verify.json", result)
    elif args.command == "certify-hosts":
        result = certify_final_hosts(
            root,
            attestation=attestation,
            qwen_snapshot=(root / args.qwen_snapshot).resolve(),
            pythia_snapshot=(root / args.pythia_snapshot).resolve(),
            device=args.device,
        )
        _write_once(final_target / "certify-hosts.json", result)
    elif args.command == "capability-matrix":
        result = run_final_matrix(
            root,
            attestation=attestation,
            qwen_snapshot=(root / args.qwen_snapshot).resolve(),
            pythia_snapshot=(root / args.pythia_snapshot).resolve(),
            device=args.device,
        )
        _write_once(final_target / "capability-matrix.json", result)
    elif args.command == "causality":
        result = final_causality(root)
        _write_once(final_target / "causality.json", result)
    elif args.command == "performance":
        result = final_performance(root)
        _write_once(final_target / "performance.json", result)
    elif args.command == "hostile-audit":
        result = final_hostile(root)
        _write_once(final_target / "hostile-audit.json", result)
    elif args.command == "verify":
        result = verify_release(release)
    elif args.command in {"cpu", "cuda"}:
        result = run_device(root, device=args.command, attestation=attestation)
    elif args.command == "quality":
        result = verify_quality(root)
        _write_once(root / RUN_ROOT / "quality.json", result)
    elif args.command == "portability":
        result = verify_portability(release)
        _write_once(root / RUN_ROOT / "portability.json", result)
    elif args.command == "report" and final_mode:
        result = final_report(root, attestation)
        _write_once(final_target / "report.json", result)
    else:
        result = build_report(root, release, attestation)
        _write_once(root / RUN_ROOT / "report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
