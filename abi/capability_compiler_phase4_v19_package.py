"""Repackage the immutable v18 tensor payload for the certified LayerCake v19 host."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from .capability_compiler_phase2_common import sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-v19-package/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _layercake_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.cake.package import build_package, load_package
    from layercake.cake.signing import key_id
    from layercake_extensions.route_isolated_prompt_span_core_v19 import (
        PROMPT_SPAN_FEATURE,
        ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
        ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
    )

    return (
        build_package,
        load_package,
        key_id,
        PROMPT_SPAN_FEATURE,
        ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_SHA256,
        ROUTE_ISOLATED_PROMPT_SPAN_CORE_V19_ABI_VERSION,
    )


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_EXACT_V18_PAYLOAD_V19_REPACKAGING"
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_model_loading_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v19 packaging governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v19 packaging binding changed: {relative}")
    return protocol, sha256_file(path)


def _source(root: Path, protocol: Mapping[str, Any]):
    layercake_root = (root / protocol["layercake_root"]).resolve()
    api = _layercake_api(layercake_root)
    public = (root / protocol["source_public_key"]).read_bytes()
    source_metadata = _json(root / protocol["source_metadata"])
    source = api[1](
        root / protocol["source_package"],
        trust_store={source_metadata["public_key"]["key_id"]: public},
        require_signature=True,
    )
    return api, source, public, source_metadata


def converted_manifest(source: Any, protocol: Mapping[str, Any], signer: str, feature: str):
    features = tuple(source.minimum_host_capabilities.get("features", ()))
    if feature in features:
        raise Phase3Error("source v18 manifest already declares v19 feature")
    provenance = dict(source.training_data_provenance)
    provenance["v19_repackaging"] = {
        "source_archive_sha256": protocol["bindings"][protocol["source_package"]],
        "source_tensor_payload_hash": source.tensor_payload_hash,
        "tensor_values_changed": False,
        "training_performed": False,
    }
    evidence = dict(source.evaluation_evidence)
    evidence["v19_host_construct"] = protocol["layercake_v19_decision"]
    evidence["v19_host_construct_sha256"] = protocol["bindings"][protocol["layercake_v19_decision"]]
    evidence["status"] = "V19_REPACKAGED_AWAITING_SAME_ARTIFACT_SCREEN"
    return replace(
        source,
        description="Exact v18 tensor payload repackaged for governed v19 prompt-span host screening",
        version="0.19.0-dev",
        abi_version=protocol["interface"],
        abi_hash=protocol["interface_sha256"],
        minimum_host_capabilities={"features": [*features, feature]},
        tensor_payload_hash="",
        package_hash="",
        training_data_provenance=provenance,
        evaluation_evidence=evidence,
        parent_version=source.version,
        signature={"algorithm": "ed25519", "key_id": signer},
    )


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    api, source, public, source_metadata = _source(root, protocol)
    _, _, key_id, feature, v19_sha, v19_version = api
    if v19_version != protocol["interface"] or v19_sha != protocol["interface_sha256"]:
        raise Phase3Error("LayerCake v19 interface identity changed")
    if source.manifest.abi_version != protocol["source_interface"]:
        raise Phase3Error("source package is not the bound v18 interface")
    if source.manifest.tensor_payload_hash != source_metadata["package"]["tensor_payload_hash"]:
        raise Phase3Error("source payload identity changed")
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    public_from_seed = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    if public_from_seed != public or key_id(public_from_seed) != source_metadata["public_key"]["key_id"]:
        raise Phase3Error("research signer changed")
    candidate = converted_manifest(source.manifest, protocol, key_id(public), feature)
    gates = {
        "source_signature_valid": source.signed,
        "source_archive_bound": source.archive_hash == protocol["bindings"][protocol["source_package"]],
        "source_payload_bound": source.manifest.tensor_payload_hash == protocol["source_payload_hash"],
        "interface_v19_bound": candidate.abi_version == v19_version and candidate.abi_hash == v19_sha,
        "architecture_unchanged": candidate.architecture == source.manifest.architecture,
        "tensor_schema_unchanged": candidate.tensor_shapes == source.manifest.tensor_shapes,
        "v19_feature_added_once": candidate.minimum_host_capabilities["features"].count(feature) == 1,
        "training_prohibited": True,
        "teacher_absent": True,
        "final_test_not_accessed": True,
    }
    return {
        "status": "PASS_V19_REPACKAGING_PREFLIGHT" if all(gates.values()) else "FAIL_V19_REPACKAGING_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "source_archive_sha256": source.archive_hash,
        "source_payload_hash": source.manifest.tensor_payload_hash,
        "tensor_count": len(source.tensors),
        "total_parameters": sum(tensor.numel() for tensor in source.tensors.values()),
        "gates": gates,
        "training_performed": False,
        "teacher_present": False,
        "final_test_accessed": False,
    }


def build(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v19 package output exists: {output}")
    check = preflight(root, protocol_path)
    if not check["status"].startswith("PASS"):
        raise Phase3Error("v19 package preflight failed")
    api, source, public, source_metadata = _source(root, protocol)
    build_package, load_package, key_id, feature, _, _ = api
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    manifest = converted_manifest(source.manifest, protocol, key_id(public), feature)
    output.mkdir(parents=True)
    package_path = output / "english_core_v19.cake"
    build_package(package_path, manifest, source.tensors, private_key=private_pem)
    loaded = load_package(package_path, trust_store={key_id(public): public}, require_signature=True)
    if loaded.manifest.tensor_payload_hash != source.manifest.tensor_payload_hash:
        raise Phase3Error("v19 tensor payload bytes changed")
    if set(loaded.tensors) != set(source.tensors) or any(
        not torch.equal(loaded.tensors[name], source.tensors[name]) for name in source.tensors
    ):
        raise Phase3Error("v19 tensor values changed")
    _write_immutable(output / "public_key.pem", public)
    metadata = {
        "format": "abi-capability-compiler-phase4-v19-package-result/1",
        "status": "PASS_EXACT_V18_PAYLOAD_REPACKAGED_FOR_V19",
        "protocol_sha256": protocol_sha,
        "source": {
            "path": protocol["source_package"],
            "archive_sha256": source.archive_hash,
            "package_hash": source.manifest.package_hash,
            "tensor_payload_hash": source.manifest.tensor_payload_hash,
        },
        "package": {
            "path": package_path.name,
            "sha256": sha256_file(package_path),
            "bytes": package_path.stat().st_size,
            "package_hash": loaded.manifest.package_hash,
            "tensor_payload_hash": loaded.manifest.tensor_payload_hash,
        },
        "public_key": {"path": "public_key.pem", "sha256": sha256_file(output / "public_key.pem"), "key_id": key_id(public)},
        "interface": loaded.manifest.abi_version,
        "interface_sha256": loaded.manifest.abi_hash,
        "tensor_count": len(loaded.tensors),
        "total_parameters": sum(tensor.numel() for tensor in loaded.tensors.values()),
        "tensor_payload_bytes_identical": loaded.manifest.tensor_payload_hash == source.manifest.tensor_payload_hash,
        "tensor_values_identical": True,
        "source_tensor_values_changed": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "training_performed": False,
        "teacher_present": False,
        "source_transformer_blocks": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Exact v18-payload-to-v19 repackaging only; output identity, prospective quality, runtime, stable frontier, matched baselines, final test, Phase 4, and superiority remain unproven.",
    }
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    command = sub.add_parser("build")
    command.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = preflight(root, root / args.protocol) if args.command == "preflight" else build(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
