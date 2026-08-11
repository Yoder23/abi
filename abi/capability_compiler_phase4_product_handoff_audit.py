"""Audit the Phase 3 composite against LayerCake's declared product handoffs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable


FORMAT = "abi-capability-compiler-phase4-product-handoff-audit/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error(f"expected JSON object: {path}")
    return value


def _tensor_inventory(path: Path) -> dict[str, Any]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
        parameters = sum(handle.get_tensor(key).numel() for key in keys)
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tensors": len(keys),
        "parameters": parameters,
        "metadata": metadata,
        "state_key_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }


def audit(root: Path, external_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = _json(protocol_path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "SEALED_READ_ONLY_PRODUCT_HANDOFF_AUDIT"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("product handoff governance changed")
    for relative, expected in protocol["abi_bindings"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"ABI binding changed: {relative}")
    for relative, expected in protocol["layercake_bindings"].items():
        path = external_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise Phase3Error(f"LayerCake binding changed: {relative}")

    phase3 = _json(root / protocol["phase3_certificate"])
    candidate = _json(external_root / protocol["layercake_candidate"])
    metadata = _json(external_root / protocol["layercake_metadata"])
    v2 = _json(external_root / protocol["interfaces"]["v2"])
    v5 = _json(external_root / protocol["interfaces"]["v5"])
    v16 = _json(external_root / protocol["interfaces"]["v16"])

    components = {
        name: _tensor_inventory(root / relative)
        for name, relative in protocol["phase3_components"].items()
    }
    product_hash = metadata["checkpoint"]["sha256"]
    product_identity_consistent = (
        product_hash == candidate["verification_summary"]["primary_checkpoint_sha256"]
        == sha256_file(external_root / protocol["layercake_product_checkpoint"])
    )
    component_formats = {
        name: value["metadata"].get("format") for name, value in components.items()
    }
    current_is_single_package = False
    v2_compatible = (
        current_is_single_package
        and v2.get("version") == "lc-direct-neural-core/2"
        and v2.get("artifact", {}).get("composition") == "direct_core_only_no_router"
    )
    v5_compatible = (
        current_is_single_package
        and v5.get("version") == "lc-direct-neural-core/5"
        and v5.get("private_representation")
        == "portable_token_plan_selective_boundary_bpe_pointer_transformer"
    )
    v16_compatible = (
        current_is_single_package
        and v16.get("artifact_compatibility", {}).get("accepted_artifact_abi")
        == component_formats.get("model")
    )
    interfaces = {
        "lc-direct-neural-core/2": {
            "compatible": v2_compatible,
            "reason": "requires one signed direct-core artifact with the v2 Unicode token plan; the Phase 3 endpoint is a three-checkpoint composite plus runtime guard",
        },
        "lc-direct-neural-core/5": {
            "compatible": v5_compatible,
            "reason": "requires one selective-boundary BPE pointer-transformer artifact; the Phase 3 model carries a different shallow-sparse state schema",
        },
        "lc-direct-neural-core/16": {
            "compatible": v16_compatible,
            "reason": "accepts an unchanged lc-direct-neural-core/15 tensor state; no Phase 3 component declares that ABI",
        },
    }
    checks = {
        "phase3_machine_evidence_complete": bool(phase3["machine_evidence_complete"]),
        "layercake_product_identity_consistent": product_identity_consistent,
        "legacy_and_product_checkpoints_distinct": protocol["legacy_checkpoint_sha256"] != product_hash,
        "phase3_endpoint_is_single_signed_package": current_is_single_package,
        "some_declared_handoff_accepts_phase3_endpoint_unchanged": any(
            value["compatible"] for value in interfaces.values()
        ),
        "training_not_performed": True,
        "final_test_not_accessed": True,
    }
    result = {
        "format": "abi-capability-compiler-phase4-product-handoff-audit-result/1",
        "status": "FAIL_NO_DECLARED_LAYERCAKE_HANDOFF_ACCEPTS_PHASE3_COMPOSITE",
        "protocol_sha256": sha256_file(protocol_path),
        "phase3_components": components,
        "phase3_component_formats": component_formats,
        "phase3_component_count": len(components),
        "phase3_component_bytes": sum(value["bytes"] for value in components.values()),
        "layercake_product": {
            "architecture": metadata["architecture"]["architecture_version"],
            "checkpoint_sha256": product_hash,
            "parameters": metadata["parameters"]["total"],
            "identity_consistent": product_identity_consistent,
        },
        "declared_handoffs": interfaces,
        "checks": checks,
        "causal_conclusion": "The successful Phase 3 endpoint is reproducible ABI evidence, but it is not one LayerCake-installable artifact and no declared LayerCake handoff accepts its component schemas unchanged. The product handoff is a direct-core replacement boundary, not an additive mutation of the sealed Phase 2 checkpoint.",
        "required_boundary": {
            "role": "english-core",
            "one_non_executable_signed_content_addressed_package": True,
            "component_schemas": component_formats,
            "teacher_absent": True,
            "receiver_training": 0,
            "artifact_immutable": True,
            "persistent_incremental_state": True,
            "strict_utf8_boundary": True,
            "runtime_guard_in_package_contract": True,
            "source_transformer_blocks": 0,
        },
        "decision": "Do not retrain or relabel the Phase 3 composite as v2, v5, or v15. Define and construct-test one new generic LayerCake handoff that packages the exact composite architecture and runtime guard without tensor mutation; then rebuild the Phase 4 frontier as self-contained artifacts against that immutable boundary.",
        "training_performed": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Read-only cross-repository compatibility result. It changes neither repository's historical evidence and proves no Phase 4, final-test, minimum-information, or superiority claim.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--layercake-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = audit(root, Path(args.layercake_root).resolve(), root / args.protocol)
    _write_immutable(root / args.output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
