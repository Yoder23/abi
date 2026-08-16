"""Run the Phase 7 product from a pre-materialized exact core archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from . import capability_compiler_phase7_integrated_runtime as base
from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase4_b20_v25_physical_screen import _api
from .capability_compiler_phase4_v19_frontier_rescreen import _json


LIFECYCLE_STATUS = "PREREGISTERED_PHASE7_DIRECT_ARTIFACT_LIFECYCLE_REPAIR"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    document = _json(path)
    base_path = (root / str(document.get("base_protocol", ""))).resolve()
    if (
        document.get("format") != base.FORMAT
        or document.get("status") != LIFECYCLE_STATUS
        or not base_path.is_file()
        or sha256_file(base_path) != document.get("base_protocol_sha256")
    ):
        raise Phase3Error("Phase 7 direct-artifact base governance changed")
    inherited = _json(base_path)
    protocol = {
        **inherited,
        **document,
        "bindings": {
            **inherited.get("bindings", {}),
            **document.get("bindings", {}),
        },
    }
    protocol_sha = sha256_file(path)
    if (
        int(protocol.get("seed", -1)) != base.SEED
        or protocol.get("devices") != ["cpu", "cuda"]
        or int(protocol.get("core_distinct_prompts", 0)) != 100
        or int(protocol.get("core_repeated_observations", 0)) != 20
        or int(protocol.get("domain_distinct_prompts", 0)) != 100
        or int(protocol.get("domain_repeated_observations", 0)) != 20
        or protocol.get("training_authorized") is not False
        or protocol.get("teacher_query_authorized") is not False
        or protocol.get("artifact_mutation_authorized") is not False
        or protocol.get("repair_of")
        != "ABI_CAPABILITY_COMPILER_PHASE7_INTEGRATED_RUNTIME_PROTOCOL_V1040.json"
        or protocol.get("preserved_failure")
        != "ABI_CAPABILITY_COMPILER_PHASE7_RSS_REPLICATION_RESULT_V1049.json"
        or protocol.get("repair_scope")
        != "DIRECT_HASH_BOUND_CORE_ARTIFACT_ACTIVATION_ONLY"
        or protocol.get("materialization_status")
        != "PASS_PHASE7_PRODUCT_MATERIALIZATION"
    ):
        raise Phase3Error("Phase 7 direct-artifact lifecycle governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(
                f"Phase 7 direct-artifact binding changed: {relative}"
            )
    archive = (root / protocol["materialized_core_archive"]).resolve()
    materialization = _json(root / protocol["materialization_result"])
    if (
        not archive.is_file()
        or sha256_file(archive) != protocol["product"]["core_archive_sha256"]
        or materialization.get("status")
        != "PASS_PHASE7_PRODUCT_MATERIALIZATION"
        or materialization.get("archive_sha256")
        != protocol["product"]["core_archive_sha256"]
        or materialization.get("tensor_payload_sha256")
        != protocol["product"]["core_payload_sha256"]
    ):
        raise Phase3Error("Phase 7 materialized core identity changed")
    return protocol, protocol_sha


def _direct_archive(
    root: Path,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], bytes, Path]:
    core_protocol = _json(root / protocol["core_protocol"])
    api = _api((root / core_protocol["layercake_root"]).resolve())
    private = Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(core_protocol["research_signing_seed_hex"])
    )
    public = private.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signer = api["key_id"](public)
    archive = (root / protocol["materialized_core_archive"]).resolve()
    materialization = _json(root / protocol["materialization_result"])
    built = {
        "archive_sha256": materialization["archive_sha256"],
        "tensor_payload_hash": materialization["tensor_payload_sha256"],
        "package_hash": materialization["package_hash"],
        "component_parameters": materialization["component_parameters"],
        "total_parameters": materialization["total_parameters"],
        "archive_bytes": materialization["archive_bytes"],
        "signer": signer,
    }
    return core_protocol, api, built, public, archive


def preflight(root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    original_loader = base.load_protocol
    base.load_protocol = lambda _root, _path: (protocol, protocol_sha)
    try:
        inherited = base.preflight(root, protocol_path)
    finally:
        base.load_protocol = original_loader
    materialization = _json(root / protocol["materialization_result"])
    gates = {
        **inherited["gates"],
        "direct_artifact_lifecycle_registered": True,
        "materialized_archive_exact": sha256_file(
            root / protocol["materialized_core_archive"]
        )
        == protocol["product"]["core_archive_sha256"],
        "materialized_payload_exact": materialization["tensor_payload_sha256"]
        == protocol["product"]["core_payload_sha256"],
        "same_process_reconstruction_absent": True,
    }
    return {
        "format": "abi-capability-compiler-phase7-direct-artifact-preflight/1",
        "status": "PASS_PHASE7_DIRECT_ARTIFACT_PREFLIGHT"
        if all(gates.values())
        else "FAIL_PHASE7_DIRECT_ARTIFACT_PREFLIGHT",
        "protocol_sha256": protocol_sha,
        "gates": gates,
    }


def run(
    root: Path,
    protocol_path: Path,
    *,
    device: str,
    output: Path,
) -> dict[str, Any]:
    protocol, _ = load_protocol(root, protocol_path)
    original_builder = base._build_core_archive
    original_writer = base._write_immutable
    original_loader = base.load_protocol

    def direct_builder(
        builder_root: Path,
        builder_protocol: Mapping[str, Any],
        _temporary: Path,
    ):
        return _direct_archive(builder_root, builder_protocol)

    def annotated_writer(path: Path, payload: bytes) -> None:
        if path.name == "result.json":
            document = json.loads(payload)
            materialization = _json(root / protocol["materialization_result"])
            document.update(
                serving_lifecycle="direct_hash_bound_materialized_archive",
                same_process_archive_reconstruction=False,
                materialized_core_archive=protocol["materialized_core_archive"],
                materialized_core_archive_sha256=protocol["product"][
                    "core_archive_sha256"
                ],
                one_time_materialization_seconds=materialization[
                    "materialization_seconds"
                ],
                package_build_seconds_field_semantics=(
                    "Legacy result field; in this repaired run it measures only "
                    "hash-bound artifact resolution, not package construction."
                ),
                preserved_reproducible_failure=protocol["preserved_failure"],
            )
            document.pop("evidence_sha256", None)
            document["evidence_sha256"] = hashlib.sha256(
                canonical_json_bytes(document)
            ).hexdigest()
            payload = json.dumps(document, indent=2, sort_keys=True).encode() + b"\n"
        original_writer(path, payload)

    base._build_core_archive = direct_builder
    base._write_immutable = annotated_writer
    base.load_protocol = lambda _root, _path: (protocol, sha256_file(protocol_path))
    try:
        base.run(root, protocol_path, device=device, output=output)
    finally:
        base._build_core_archive = original_builder
        base._write_immutable = original_writer
        base.load_protocol = original_loader
    return _json(output / "result.json")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    protocol_path = (root / args.protocol).resolve()
    if args.preflight:
        result = preflight(root, protocol_path)
    elif args.device and args.output_dir:
        result = run(
            root,
            protocol_path,
            device=args.device,
            output=(root / args.output_dir).resolve(),
        )
    else:
        raise Phase3Error("select preflight or one device and output directory")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
