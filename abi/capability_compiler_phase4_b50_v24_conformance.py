"""Repackage exact B50 tensors for LayerCake v24 and prove conformance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import torch

from . import capability_compiler_phase4_b50_v23_conformance as v23_engine
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b50-v24-conformance/1"
RESULT_FORMAT = "abi-capability-compiler-phase4-b50-v24-conformance-result/1"
_V23_API = v23_engine._api_v23
_V23_REPACKAGE = v23_engine._repackage


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    overlay = _json(path)
    if (
        overlay.get("format") != FORMAT
        or overlay.get("status")
        != "PREREGISTERED_EXACT_B50_V24_REPACKAGE_AND_CONFORMANCE"
        or overlay.get("device") != "cuda"
        or overlay.get("training_authorized") is not False
        or overlay.get("teacher_model_loading_authorized") is not False
        or overlay.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("exact B50 v24 conformance governance changed")
    for relative, expected in overlay["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"exact B50 v24 conformance binding changed: {relative}")
    base, _ = v23_engine.load_protocol(
        root, root / str(overlay["base_conformance_protocol"])
    )
    merged = dict(base)
    merged["runtime_interface"] = "lc-direct-neural-core/24"
    return merged, sha256_file(path)


def _api_v24(layercake_root: Path) -> dict[str, Any]:
    api = _V23_API(layercake_root)
    from layercake_extensions.route_isolated_allocation_bounded_core_v24 import (
        ALLOCATION_BOUNDED_ADOPTION_FEATURE,
        ARCHITECTURE_V24_FORMAT,
        ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256,
        ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION,
        AllocationBoundedRuntimeResidencyCoreHost,
    )

    return {
        **api,
        "architecture_format": ARCHITECTURE_V24_FORMAT,
        "Host": AllocationBoundedRuntimeResidencyCoreHost,
        "abi_sha256": ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_SHA256,
        "abi_version": ROUTE_ISOLATED_ALLOCATION_BOUNDED_CORE_V24_ABI_VERSION,
        "allocation_feature": ALLOCATION_BOUNDED_ADOPTION_FEATURE,
    }


def _repackage_v24(
    root: Path,
    source_protocol: Mapping[str, Any],
    spec: Mapping[str, Any],
    directory: Path,
    api_v22: Mapping[str, Any],
    api_v24: Mapping[str, Any],
    private: Ed25519PrivateKey,
    public_pem: bytes,
) -> tuple[Path, dict[str, Any]]:
    layercake_root = (root / str(source_protocol["layercake_root"])).resolve()
    api_v23 = _V23_API(layercake_root)
    v23_path, parent = _V23_REPACKAGE(
        root,
        source_protocol,
        spec,
        directory,
        api_v22,
        api_v23,
        private,
        public_pem,
    )
    signer = api_v24["key_id"](public_pem)
    loaded_v23 = api_v23["load_package"](
        v23_path, trust_store={signer: public_pem}, require_signature=True
    )
    document = loaded_v23.manifest.canonical_dict()
    features = list(document["minimum_host_capabilities"]["features"])
    features.append(str(api_v24["allocation_feature"]))
    seed = int(spec["seed"])
    document.update(
        {
            "cake_id": f"abi-phase4-v24-b50-seed{seed}-english-core",
            "name": f"ABI Phase 4 v24 B50 seed {seed} English core",
            "description": "Exact v23 B50 tensors on isolated allocation-bounded v24 host",
            "version": "0.24.0-b50-allocation-bounded",
            "abi_version": api_v24["abi_version"],
            "abi_hash": api_v24["abi_sha256"],
            "minimum_host_capabilities": {"features": features},
            "tensor_payload_hash": "",
            "package_hash": "",
            "evaluation_evidence": {
                "status": "V24_EXACT_TENSOR_REPACKAGE_OUTPUT_CONFORMANCE",
                "parent_tensor_payload_hash": loaded_v23.manifest.tensor_payload_hash,
            },
        }
    )
    manifest = api_v24["CakeManifest"].from_dict(document)
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    path = directory / "candidate-v24.cake"
    api_v24["build_package"](
        path, manifest, loaded_v23.tensors, private_key=private_pem
    )
    loaded_v24 = api_v24["load_package"](
        path, trust_store={signer: public_pem}, require_signature=True
    )
    tensors_exact = set(loaded_v24.tensors) == set(loaded_v23.tensors) and all(
        torch.equal(loaded_v24.tensors[name], loaded_v23.tensors[name])
        for name in loaded_v23.tensors
    )
    gates = {
        "signature_valid": loaded_v24.signed,
        "v24_interface": loaded_v24.manifest.abi_version == api_v24["abi_version"]
        and loaded_v24.manifest.abi_hash == api_v24["abi_sha256"],
        "tensor_values_exact_to_v23": tensors_exact,
        "tensor_payload_hash_exact_to_v23": loaded_v24.manifest.tensor_payload_hash
        == loaded_v23.manifest.tensor_payload_hash
        == parent["tensor_payload_hash"],
        "single_parse_feature_declared": api_v24["single_parse_feature"]
        in loaded_v24.manifest.minimum_host_capabilities["features"],
        "allocation_feature_declared": api_v24["allocation_feature"]
        in loaded_v24.manifest.minimum_host_capabilities["features"],
    }
    if not all(gates.values()):
        raise Phase3Error(f"exact B50 v24 repackage failed: {gates}")
    return path, {
        "archive_sha256": loaded_v24.archive_hash,
        "archive_bytes": path.stat().st_size,
        "package_hash": loaded_v24.manifest.package_hash,
        "tensor_payload_hash": loaded_v24.manifest.tensor_payload_hash,
        "parent_v23_archive_sha256": parent["archive_sha256"],
        "parent_v22_archive_sha256": parent["parent_v22_archive_sha256"],
        "component_parameters": parent["component_parameters"],
        "total_parameters": parent["total_parameters"],
        "tensor_count": parent["tensor_count"],
        "gates": gates,
    }


def run(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable exact B50 v24 output exists: {output}")
    original_loader = v23_engine.load_protocol
    original_api = v23_engine._api_v23
    original_repackage = v23_engine._repackage
    v23_engine.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    v23_engine._api_v23 = _api_v24
    v23_engine._repackage = _repackage_v24
    try:
        engine = v23_engine.run(root, protocol_path, output / "engine")
    finally:
        v23_engine.load_protocol = original_loader
        v23_engine._api_v23 = original_api
        v23_engine._repackage = original_repackage
    adoption = all(
        int(system["activation"]["strict_assigned_tensor_count"])
        == int(system["activation"]["authenticated_tensor_count"])
        == int(system["package"]["tensor_count"])
        and int(system["activation"]["meta_tensors_after_adoption"]) == 0
        for system in engine["systems"]
    )
    passed = engine["status"].startswith("PASS") and adoption
    engine_path = output / "engine" / "result.json"
    aggregate = output / "engine" / "all_outputs.jsonl"
    result = {
        "format": RESULT_FORMAT,
        "status": "PASS_EXACT_B50_V24_THREE_SEED_OUTPUT_CONFORMANCE"
        if passed
        else "FAIL_EXACT_B50_V24_OUTPUT_CONFORMANCE",
        "protocol_sha256": protocol_sha,
        "runtime_interface": "lc-direct-neural-core/24",
        "engine_status": engine["status"],
        "engine_result": engine_path.relative_to(root).as_posix(),
        "engine_result_sha256": sha256_file(engine_path),
        "aggregate_outputs": aggregate.relative_to(root).as_posix(),
        "aggregate_outputs_sha256": sha256_file(aggregate),
        "systems": engine["systems"],
        "observations": engine["observations"],
        "strict_tensor_adoption": adoption,
        "training_performed": False,
        "teacher_model_loaded": False,
        "receiver_training_steps": 0,
        "receiver_calibration_runs": 0,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Exact three-seed v23-to-v24 tensor and development-output conformance only. No training, teacher query, final test, runtime, Phase 4, or ABI-superiority claim.",
    }
    if not result_evidence_digest_valid(engine):
        raise Phase3Error("exact B50 v24 engine evidence digest changed")
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(
        output / "result.json",
        json.dumps(result, indent=2, sort_keys=True).encode() + b"\n",
    )
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = run(root, root / args.protocol, root / args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
