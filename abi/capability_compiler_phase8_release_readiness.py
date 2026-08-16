"""Build and verify the local Phase 8 content-addressed handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping

import psutil
import torch

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_v19_frontier_rescreen import _json
from .capability_compiler_phase7_direct_artifact_runtime import (
    load_protocol as load_product_protocol,
)


FORMAT = "abi-capability-compiler-phase8-release-readiness/1"
RESULT_FORMAT = "abi-capability-compiler-phase8-release-readiness-result/1"


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def hardware_document() -> dict[str, Any]:
    cuda = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda = {
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "torch_cuda_version": torch.version.cuda,
        }
    document = {
        "format": "abi-capability-compiler-hardware-fingerprint/1",
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine_architecture": platform.machine(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "total_ram_bytes": int(psutil.virtual_memory().total),
        "cuda": cuda,
    }
    document["fingerprint_sha256"] = hashlib.sha256(
        canonical_json_bytes(document)
    ).hexdigest()
    return document


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_PHASE8_LOCAL_RELEASE_READINESS"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("external_self_attestation_authorized") is not False
    ):
        raise Phase3Error("Phase 8 release-readiness governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 8 release binding changed: {relative}")
    return protocol, sha256_file(path)


def _release_inventory(
    root: Path, protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    product, _ = load_product_protocol(root, root / protocol["product_protocol"])
    base = _json(root / product["base_protocol"])
    bindings = {**base.get("bindings", {}), **product.get("bindings", {})}
    base_runtime = _json(root / product["base_runtime_protocol"])
    bindings[str(base_runtime["development_catalog"])] = sha256_file(
        root / base_runtime["development_catalog"]
    )
    for relative in protocol["additional_release_files"]:
        bindings[relative] = protocol["bindings"][relative]
    inventory: dict[str, dict[str, Any]] = {}
    for relative, expected in sorted(bindings.items()):
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"Phase 8 release inventory changed: {relative}")
        inventory[relative] = {
            "sha256": expected,
            "bytes": target.stat().st_size,
            "repository": "layercake" if relative.startswith("../layercake_release/") else "abi",
            "tracked": bool(
                subprocess.run(
                    ["git", "ls-files", "--error-unmatch", "--", relative],
                    cwd=root,
                    capture_output=True,
                    text=True,
                ).returncode
                == 0
            )
            if not relative.startswith("../layercake_release/")
            else True,
        }
    return inventory


def validate_manifest_document(
    root: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, bool]:
    expected = _release_inventory(root, protocol)
    declared = manifest.get("files", {})
    recomputed_bytes = sum(int(row["bytes"]) for row in expected.values())
    return {
        "format_exact": manifest.get("format") == RESULT_FORMAT,
        "local_status_exact": manifest.get("status")
        == "PASS_PHASE8_LOCAL_RELEASE_READINESS",
        "file_inventory_exact": declared == expected,
        "file_count_exact": int(manifest.get("file_count", -1)) == len(expected),
        "byte_count_exact": int(manifest.get("total_bytes", -1))
        == recomputed_bytes,
        "abi_phase7_seal_ancestor": manifest.get("source", {}).get(
            "abi_phase7_seal_commit"
        )
        == protocol["abi_phase7_seal_commit"],
        "layercake_commit_exact": manifest.get("source", {}).get(
            "layercake_commit"
        )
        == protocol["layercake_commit"],
        "development_hardware_bound": manifest.get("development_hardware", {}).get(
            "fingerprint_sha256"
        )
        == protocol["development_hardware_fingerprint_sha256"],
        "external_operator_not_self_attested": manifest.get("external_gates", {}).get(
            "independent_operator_complete"
        )
        is False,
        "external_hardware_not_self_attested": manifest.get("external_gates", {}).get(
            "independent_hardware_complete"
        )
        is False,
        "phase8_not_certified_locally": manifest.get("phase8_certified") is False,
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    abi_head = _git_head(root)
    layercake_head = _git_head((root / "../layercake_release").resolve())
    development = hardware_document()
    inventory = _release_inventory(root, protocol)
    gates = {
        "abi_phase7_seal_is_ancestor_or_head": subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                protocol["abi_phase7_seal_commit"],
                abi_head,
            ],
            cwd=root,
        ).returncode
        == 0,
        "layercake_commit_exact": layercake_head == protocol["layercake_commit"],
        "development_hardware_exact": development["fingerprint_sha256"]
        == protocol["development_hardware_fingerprint_sha256"],
        "release_inventory_nonempty": len(inventory) >= 40,
        "output_absent": not (root / protocol["output"]).exists(),
        "adversarial_output_absent": not (
            root / protocol["adversarial_output"]
        ).exists(),
        "model_inference_absent": True,
        "training_absent": True,
        "external_self_attestation_absent": True,
    }
    return {
        "format": "abi-capability-compiler-phase8-release-readiness-preflight/1",
        "status": "PASS_PHASE8_RELEASE_READINESS_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE8_RELEASE_READINESS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "abi_head": abi_head,
        "layercake_head": layercake_head,
        "development_hardware": development,
        "release_file_count": len(inventory),
        "release_total_bytes": sum(int(row["bytes"]) for row in inventory.values()),
        "gates": gates,
    }


def build(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error("immutable Phase 8 readiness manifest exists")
    inventory = _release_inventory(root, protocol)
    development = hardware_document()
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_PHASE8_LOCAL_RELEASE_READINESS",
        "protocol_sha256": protocol_sha,
        "source": {
            "abi_phase7_seal_commit": protocol["abi_phase7_seal_commit"],
            "abi_packet_source_commit": _git_head(root),
            "layercake_commit": _git_head((root / "../layercake_release").resolve()),
        },
        "development_hardware": development,
        "files": inventory,
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory.values()),
        "external_commands": protocol["external_commands"],
        "required_external_records": protocol["required_external_records"],
        "external_gates": {
            "independent_operator_complete": False,
            "independent_hardware_complete": False,
            "fresh_cpu_result_complete": False,
            "fresh_cuda_result_complete": False,
            "external_hostile_verification_complete": False,
        },
        "model_inference_performed": False,
        "training_performed": False,
        "phase8_certified": False,
        "blocked_on": [
            "independent human operator",
            "independent CPU and CUDA hardware",
            "publication or transfer of the two exact repository commits and untracked content-addressed payloads",
        ],
        "claim_boundary": "Local content-addressed handoff readiness only. This manifest deliberately records every external gate as incomplete and cannot certify Phase 8 or release.",
    }
    gates = validate_manifest_document(root, protocol, result)
    result["gates"] = gates
    result["status"] = (
        "PASS_PHASE8_LOCAL_RELEASE_READINESS"
        if all(gates.values())
        else "FAIL_PHASE8_LOCAL_RELEASE_READINESS"
    )
    result["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(result)
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output,
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def verify_manifest(
    root: Path, protocol_path: Path, manifest_path: Path
) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    manifest = _json(manifest_path)
    gates = validate_manifest_document(root, protocol, manifest)
    gates.update(
        manifest_protocol_exact=manifest.get("protocol_sha256") == protocol_sha,
        manifest_evidence_hash=(
            lambda document: document.pop("evidence_sha256", None)
            == hashlib.sha256(canonical_json_bytes(document)).hexdigest()
        )(dict(manifest)),
        layercake_head_exact=_git_head((root / "../layercake_release").resolve())
        == protocol["layercake_commit"],
    )
    return {
        "format": "abi-capability-compiler-phase8-release-manifest-verification/1",
        "status": "PASS_PHASE8_RELEASE_MANIFEST_VERIFICATION"
        if all(gates.values())
        else "FAIL_PHASE8_RELEASE_MANIFEST_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "manifest_sha256": sha256_file(manifest_path),
        "gates": gates,
        "phase8_certified": False,
    }


def capture_hardware(output: Path) -> dict[str, Any]:
    if output.exists():
        raise Phase3Error("immutable hardware record exists")
    document = hardware_document()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output,
        json.dumps(document, indent=2, sort_keys=True).encode() + b"\n",
    )
    return document


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--verify-manifest")
    parser.add_argument("--capture-hardware")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.capture_hardware:
        result = capture_hardware((root / args.capture_hardware).resolve())
        result = {"status": "PASS_HARDWARE_CAPTURE", **result}
    elif args.preflight:
        result = preflight(root, protocol_path)
    elif args.verify_manifest:
        result = verify_manifest(
            root, protocol_path, (root / args.verify_manifest).resolve()
        )
    elif args.output:
        result = build(root, protocol_path, (root / args.output).resolve())
    else:
        raise Phase3Error("select preflight or output")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
