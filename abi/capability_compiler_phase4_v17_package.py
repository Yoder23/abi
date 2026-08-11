"""Package the exact successful Phase 3 composite for LayerCake v17."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from safetensors.torch import load_file
import torch

from .capability_compiler_phase2_common import CAPABILITIES, sha256_file
from .capability_compiler_phase3 import CAPABILITY_TO_ROUTE, Phase3Error, _write_immutable
from .capability_compiler_phase3_guarded_screen import artifact_markers
from .capability_compiler_phase3_weak_residual import WEAK_CAPABILITIES


FORMAT = "abi-capability-compiler-phase4-v17-package/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "SEALED_EXACT_PHASE3_COMPOSITE_PACKAGING"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("v17 package governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"v17 package binding changed: {relative}")
    return protocol, sha256_file(path)


def _states(root: Path, protocol: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: load_file(str((root / relative).resolve()), device="cpu")
        for name, relative in protocol["components"].items()
    }


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol(root, protocol_path)
    states = _states(root, protocol)
    counts = {
        name: sum(tensor.numel() for tensor in state.values())
        for name, state in states.items()
    }
    expected = {"model": 61655050, "router": 1058040, "residual": 99840}
    if counts != expected:
        raise Phase3Error(f"v17 component parameter counts changed: {counts}")
    namespaced = {
        f"{namespace}.{name}": tensor
        for namespace, state in states.items()
        for name, tensor in state.items()
    }
    namespace_counts = Counter(name.split(".", 1)[0] for name in namespaced)
    if namespace_counts != Counter({"model": 82, "router": 3, "residual": 4}):
        raise Phase3Error("v17 tensor namespace inventory changed")
    return {
        "status": "PASS_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "component_parameters": counts,
        "total_parameters": sum(counts.values()),
        "tensor_namespaces": dict(namespace_counts),
        "teacher_loaded": False,
        "training_performed": False,
        "final_test_accessed": False,
    }


def _layercake_api(layercake_root: Path):
    if str(layercake_root) not in sys.path:
        sys.path.insert(0, str(layercake_root))
    from layercake.cake.manifest import CakeManifest
    from layercake.cake.package import build_package, load_package, tensor_specs
    from layercake.cake.signing import key_id
    from layercake_extensions.route_isolated_shallow_sparse_core import (
        ARCHITECTURE_FORMAT,
        CAPABILITY_TO_TASK_ROUTE,
        ROUTE_ISOLATED_CORE_ABI_SHA256,
        ROUTE_ISOLATED_CORE_ABI_VERSION,
    )
    return CakeManifest, build_package, load_package, tensor_specs, key_id, ARCHITECTURE_FORMAT, CAPABILITY_TO_TASK_ROUTE, ROUTE_ISOLATED_CORE_ABI_SHA256, ROUTE_ISOLATED_CORE_ABI_VERSION


def build(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable v17 package output exists: {output}")
    check = preflight(root, protocol_path)
    layercake_root = (root / protocol["layercake_root"]).resolve()
    CakeManifest, build_package, load_package, tensor_specs, key_id, architecture_format, task_routes, abi_sha, abi_version = _layercake_api(layercake_root)
    if abi_version != protocol["interface"] or abi_sha != protocol["interface_sha256"]:
        raise Phase3Error("v17 LayerCake interface identity changed")
    states = _states(root, protocol)
    tensors = {
        f"{namespace}.{name}": tensor
        for namespace, state in states.items()
        for name, tensor in state.items()
    }
    parent = _json(root / protocol["model_metadata"])
    model_tokenizer = _json(root / protocol["model_tokenizer"])
    model_tokenizer_raw = json.dumps(model_tokenizer, sort_keys=True, separators=(",", ":")).encode()
    router_tokenizer = _json(root / protocol["router_tokenizer"])
    router_config = _json(root / protocol["router_config"])
    markers = artifact_markers(root / protocol["guard_artifact"])
    architecture = {
        "format": architecture_format,
        "model": parent["architecture"],
        "model_tokenizer": {
            "format": "declarative-tokenizers-json/1",
            "tokenizers_json": model_tokenizer,
            "sha256": hashlib.sha256(model_tokenizer_raw).hexdigest(),
            "eos_token_id": 50256,
        },
        "router": {
            "vocabulary": int(router_config["vocabulary"]),
            "character_hash_buckets": int(router_config["character_hash_buckets"]),
            "character_ngram_minimum": int(router_config["character_ngram_minimum"]),
            "character_ngram_maximum": int(router_config["character_ngram_maximum"]),
            "hash_seed": int(router_config["hash_seed"]),
            "classes": len(CAPABILITIES) + 1,
        },
        "router_tokenizer": router_tokenizer,
        "residual": {"width": 768, "rank": 16, "routes": 4, "reuse": "before_each_transformer_block"},
        "capabilities": list(CAPABILITIES),
        "capability_to_task_route": task_routes,
        "weak_capabilities": list(WEAK_CAPABILITIES),
        "guard": {
            "predicate": "contiguous_1_to_16_token_span_repeated_4_times_or_fourgram_diversity_below_0.35_at_32_tokens",
            "scope": "weak_capabilities_only",
            "stop_before_collapsing_token": True,
            "abstention_markers": list(markers),
            "abstention_clause": "I cannot determine that from the information given.",
        },
    }
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(protocol["research_signing_seed_hex"]))
    private_pem = private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public_pem = private.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    signer = key_id(public_pem)
    manifest = CakeManifest(
        schema_version="1",
        cake_id="abi-phase3-route-isolated-english-core",
        name="ABI route-isolated English core",
        description="Exact packaged Phase 3 development composite for governed conformance",
        version="0.17.0-dev",
        publisher={"id": "abi-research", "name": "ABI Research", "key_id": signer},
        abi_version=abi_version,
        abi_hash=abi_sha,
        cake_type="portable_decoder",
        input_contract={"external": "UTF-8 bytes", "role": "english-core", "validity": "strict_utf8"},
        output_contract={"external": "UTF-8 bytes", "role": "english-core", "composition": "direct_core_only_no_router", "validity": "strict_utf8"},
        architecture=architecture,
        supported_precisions=("fp32",),
        supported_backends=("pytorch", "cuda"),
        minimum_host_capabilities={"features": ["byte_input", "safe_tensors", "persistent_incremental_state", "physical_route_isolation", "declarative_runtime_guard", "strict_utf8_boundary"]},
        tensor_payload_hash="",
        tensor_shapes=tensor_specs(tensors),
        package_hash="",
        training_data_provenance={"phase1_ir_sha256": protocol["phase1_ir_sha256"], "component_sha256": {name: protocol["bindings"][relative] for name, relative in protocol["components"].items()}, "teacher_at_inference": False, "source_transformer_blocks": 0},
        evaluation_evidence={"phase3_machine_audit": protocol["phase3_certificate"], "phase3_machine_audit_sha256": protocol["bindings"][protocol["phase3_certificate"]], "status": "DEVELOPMENT_MACHINE_EVIDENCE_ONLY"},
        license="Apache-2.0",
        dependencies=(),
        parent_version=None,
        signature={"algorithm": "ed25519", "key_id": signer},
        domains=("english-core",),
        permissions=("local-inference",),
    )
    output.mkdir(parents=True)
    package_path = output / "english_core_v17.cake"
    build_package(package_path, manifest, tensors, private_key=private_pem)
    loaded = load_package(package_path, trust_store={signer: public_pem}, require_signature=True)
    _write_immutable(output / "public_key.pem", public_pem)
    metadata = {
        "format": "abi-capability-compiler-phase4-v17-package-result/1",
        "status": "PASS_EXACT_COMPOSITE_PACKAGED",
        "protocol_sha256": protocol_sha,
        "interface": abi_version,
        "interface_sha256": abi_sha,
        "package": {"path": package_path.name, "sha256": sha256_file(package_path), "bytes": package_path.stat().st_size, "package_hash": loaded.manifest.package_hash, "tensor_payload_hash": loaded.manifest.tensor_payload_hash},
        "public_key": {"path": "public_key.pem", "sha256": sha256_file(output / "public_key.pem"), "key_id": signer},
        "components": {name: {"source_path": relative, "source_sha256": protocol["bindings"][relative], "parameters": check["component_parameters"][name]} for name, relative in protocol["components"].items()},
        "tensor_namespaces": check["tensor_namespaces"],
        "total_parameters": check["total_parameters"],
        "source_tensor_values_changed": False,
        "teacher_present": False,
        "source_transformer_blocks": 0,
        "receiver_training_steps": 0,
        "training_performed": False,
        "final_test_accessed": False,
        "quality_tested": False,
        "phase4_certified": False,
        "claim_boundary": "Exact development composite packaging only; output identity, quality, information minimum, final test, Phase 4, and superiority remain unproven.",
    }
    _write_immutable(output / "metadata.json", json.dumps(metadata, indent=2, sort_keys=True).encode() + b"\n")
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve(); protocol = root / args.protocol
    result = preflight(root, protocol) if args.command == "preflight" else build(root, protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
