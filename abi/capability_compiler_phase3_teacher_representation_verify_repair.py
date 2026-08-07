"""V61 repaired hostile verification using exact extraction-batch reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable, load_phase1_ir
from .capability_compiler_phase3_teacher_representation_verify import (
    ARTIFACT_FORMAT,
    _load_jsonl,
    _verify_evidence_hash,
    _verify_records,
    _verify_tensor_structure,
)


FORMAT = "abi-capability-compiler-phase3-teacher-representation-verifier-repair/1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Phase3Error("expected JSON object")
    return value


def _load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status") != "PREREGISTERED_VERIFIER_REPAIR"
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("teacher substrate verifier repair governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"teacher substrate verifier repair binding changed: {relative}")
    return protocol, sha256_file(path)


def _verify_numerics_result(value: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    _verify_evidence_hash(value)
    original = value.get("original_batch_recomputation", {})
    singleton = value.get("singleton_recomputation", {})
    if (
        value.get("status") != "DIAGNOSTIC_COMPLETE_NONPROMOTIONAL"
        or value.get("artifact_verified") is not False
        or value.get("training_performed") is not False
        or int(original.get("vectors", -1)) != int(expected["sample_vectors"])
        or float(original.get("maximum_absolute_error", -1)) != 0.0
        or float(original.get("mean_absolute_error", -1)) != 0.0
        or float(original.get("mean_exact_scalar_fraction", -1)) != 1.0
        or float(original.get("minimum_cosine_similarity", 0)) < float(expected["minimum_cosine_similarity"])
        or float(singleton.get("maximum_absolute_error", 0)) <= float(expected["failed_singleton_threshold"])
    ):
        raise Phase3Error("V60 does not prove exact original-batch reproduction")


def verify(root: Path, protocol_path: Path, output_path: Path) -> dict[str, Any]:
    protocol, protocol_sha = _load_protocol(root, protocol_path)
    artifact_dir = (root / protocol["artifact"]["directory"]).resolve()
    metadata_path = artifact_dir / "metadata.json"
    metadata = _json(metadata_path)
    if metadata.get("format") != ARTIFACT_FORMAT or metadata.get("status") != "EXTRACTED_UNVERIFIED_TRAINING_PROHIBITED":
        raise Phase3Error("teacher substrate metadata governance changed")
    _verify_evidence_hash(metadata)
    tensor_path = artifact_dir / metadata["artifact"]["path"]
    records_path = artifact_dir / metadata["records"]["path"]
    for path, expected_hash, expected_bytes in (
        (tensor_path, protocol["artifact"]["tensor_sha256"], protocol["artifact"]["tensor_file_bytes"]),
        (records_path, protocol["artifact"]["records_sha256"], protocol["artifact"]["records_file_bytes"]),
        (metadata_path, protocol["artifact"]["metadata_sha256"], protocol["artifact"]["metadata_file_bytes"]),
    ):
        if not path.is_file() or sha256_file(path) != expected_hash or path.stat().st_size != int(expected_bytes):
            raise Phase3Error(f"teacher substrate file changed: {path.name}")
    tensors = load_file(str(tensor_path), device="cpu")
    structure = _verify_tensor_structure(tensors, int(protocol["expected"]["records"]), int(protocol["expected"]["hidden_width"]))
    phase1_rows = load_phase1_ir((root / protocol["phase1_ir"]).resolve())
    provenance = _verify_records(phase1_rows, _load_jsonl(records_path))
    if provenance["record_order_sha256"] != protocol["expected"]["record_order_sha256"]:
        raise Phase3Error("teacher substrate record order changed")
    accounting = metadata["imported_information"]
    if (
        accounting.get("hidden_activation_bytes") != int(protocol["expected"]["tensor_payload_bytes"])
        or accounting.get("logits_stored") != 0
        or accounting.get("source_parameters_copied") != 0
        or metadata.get("teacher_present_in_artifact") is not False
        or metadata.get("training_performed") is not False
        or metadata.get("final_test_accessed") is not False
    ):
        raise Phase3Error("teacher substrate imported-information boundary changed")
    numerics_path = (root / protocol["numerics"]["path"]).resolve()
    numerics = _json(numerics_path)
    _verify_numerics_result(numerics, protocol["numerics"])
    result = {
        "format": "abi-capability-compiler-phase3-teacher-representation-verification-result/2",
        "status": "PASS_ARTIFACT_VERIFIED_TRAINING_PROTOCOL_DESIGN_AUTHORIZED",
        "protocol_sha256": protocol_sha,
        "failed_predecessor_preserved": "ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_VERIFIER_FAILURE_V59.json",
        "repair": "Replace batch-shape-sensitive singleton equality with exact reproduction under the immutable original extraction batch partition; retain singleton discrepancy as evidence.",
        "artifact": {
            "metadata_sha256": sha256_file(metadata_path),
            "tensor_sha256": sha256_file(tensor_path),
            "records_sha256": sha256_file(records_path),
            "tensor_payload_bytes": int(protocol["expected"]["tensor_payload_bytes"]),
        },
        "tensor_structure": structure,
        "provenance": provenance,
        "numerical_reproduction": numerics["original_batch_recomputation"],
        "singleton_batch_shape_sensitivity": numerics["singleton_recomputation"],
        "source_manifest_sha256": numerics["source_manifest_sha256"],
        "stored_logits": 0,
        "copied_source_parameters": 0,
        "teacher_present_in_artifact": False,
        "training_performed": False,
        "layercake_host_changed": False,
        "phase3_certified": False,
        "phase4_open": False,
        "final_test_accessed": False,
        "next_authorized_work": "Design and preregister one representation-aligned LayerCake candidate and matched same-size control. Training remains unauthorized until that protocol is sealed.",
        "claim_boundary": "V61 verifies the substrate only. It does not prove learned transfer, generation quality, host performance, Phase 3 completion, or superiority."
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    _write_immutable(output_path, json.dumps(result, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_TEACHER_REPRESENTATION_VERIFIER_REPAIR_V61.json")
    parser.add_argument("--output", default="results/abi_capability_compiler_phase3_teacher_representation/verification_v61.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    print(json.dumps(verify(root, (root / args.protocol).resolve(), (root / args.output).resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
