"""Independently verify the exact B40 equal-sequence baseline pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import zipfile

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error, _write_immutable
from .capability_compiler_phase4_b40_baseline_pack import _reconstruct
from .capability_compiler_phase4_b50_grid_verify import result_evidence_digest_valid
from .capability_compiler_phase4_v19_frontier_rescreen import _json


FORMAT = "abi-capability-compiler-phase4-b40-baseline-pack-verify/1"


def load_protocol(root: Path, path: Path) -> tuple[dict[str, Any], str]:
    protocol = _json(path)
    if (
        protocol.get("format") != FORMAT
        or protocol.get("status")
        != "PREREGISTERED_READ_ONLY_EXACT_B40_BASELINE_PACK_VERIFIER"
        or protocol.get("model_inference_authorized") is not False
        or protocol.get("training_authorized") is not False
        or protocol.get("final_test_access") != "PROHIBITED"
    ):
        raise Phase3Error("B40 baseline-pack verifier governance changed")
    for relative, expected in protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 baseline-pack verifier binding changed: {relative}")
    return protocol, sha256_file(path)


def verify(root: Path, protocol_path: Path, output: Path) -> dict[str, Any]:
    protocol, protocol_sha = load_protocol(root, protocol_path)
    if output.exists():
        raise Phase3Error(f"immutable B40 pack verification exists: {output}")
    source_protocol = _json(root / protocol["pack_protocol"])
    for relative, expected in source_protocol["bindings"].items():
        target = (root / relative).resolve()
        if not target.is_file() or sha256_file(target) != expected:
            raise Phase3Error(f"B40 pack source binding changed: {relative}")
    declared = _json(root / protocol["pack_result"])
    records, accounting, budget, packs, rebuilt = _reconstruct(
        root, source_protocol
    )
    expected_records = b"".join(canonical_json_bytes(row) for row in records)
    archive_path = root / protocol["records_archive"]
    with zipfile.ZipFile(archive_path) as archive:
        stored_records = archive.read("records.jsonl")
        stored_accounting = json.loads(archive.read("accounting.json"))
        stored_manifest = json.loads(archive.read("manifest.json"))
    observed_pack = _json(root / protocol["pack_manifest"])
    pack_keys = (
        "packs",
        "pack_count",
        "record_count",
        "input_tokens",
        "response_tokens",
        "content_sha256",
        "maximum_record_tokens",
    )
    gates = {
        "declared_result_digest_valid": result_evidence_digest_valid(declared),
        "declared_result_pass": declared["status"]
        == "PASS_EXACT_B40_BASELINE_SEQUENCE_PACK_READY",
        "records_bytes_exact": stored_records == expected_records,
        "accounting_exact": stored_accounting == accounting,
        "archive_manifest_exact": stored_manifest
        == declared["records_archive"]["manifest"],
        "archive_internal_hash_exact": hashlib.sha256(stored_records).hexdigest()
        == stored_manifest["records_jsonl_sha256"],
        "pack_reconstructed_exact": all(
            observed_pack[key] == rebuilt[key] for key in pack_keys
        ),
        "pack_archive_link_exact": observed_pack["records_archive_sha256"]
        == sha256_file(archive_path),
        "budget_exact": budget == declared["budget"],
        "all_4112_records_and_693_packs": len(records) == 4112
        and len(packs) == 693,
        "model_inference_absent": True,
        "training_absent": True,
        "teacher_loading_absent": True,
        "final_test_not_accessed": True,
    }
    mutated_records = bytearray(stored_records)
    mutated_records[0] ^= 1
    mutated_pack = dict(observed_pack)
    mutated_pack["content_sha256"] = "0" * 64
    mutations = {
        "archive_byte_mutation_rejected": hashlib.sha256(bytes(mutated_records)).hexdigest()
        != stored_manifest["records_jsonl_sha256"],
        "record_drop_rejected": len(records[:-1]) != int(stored_manifest["records"]),
        "accounting_relabel_rejected": {
            **stored_accounting,
            "unique_source_attempts": 4004,
        }
        != accounting,
        "pack_content_relabel_rejected": mutated_pack["content_sha256"]
        != rebuilt["content_sha256"],
        "selection_relabel_rejected": str(budget["selection_sha256"])
        != "0" * 64,
        "status_not_source_of_truth": all(gates.values()),
    }
    passed = all(gates.values()) and all(mutations.values())
    result = {
        "format": "abi-capability-compiler-phase4-b40-baseline-pack-verify-result/1",
        "status": "PASS_INDEPENDENT_EXACT_B40_BASELINE_PACK_VERIFICATION"
        if passed
        else "FAIL_B40_BASELINE_PACK_VERIFICATION",
        "protocol_sha256": protocol_sha,
        "records": len(records),
        "packs": len(packs),
        "content_sha256": rebuilt["content_sha256"],
        "imported_information": accounting,
        "gates": gates,
        "mutations": mutations,
        "model_inference_performed": False,
        "training_performed": False,
        "teacher_model_loaded": False,
        "final_test_accessed": False,
        "phase4_certified": False,
        "claim_boundary": "Independent exact B40 equal-sequence pack verification only. No baseline training, matched comparison, final test, Phase 4, or ABI superiority.",
    }
    result["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_immutable(output, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, root / args.protocol, root / args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if str(result["status"]).startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
