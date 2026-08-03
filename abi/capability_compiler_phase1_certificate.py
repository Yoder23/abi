"""Verify the sealed capability-compiler Phase 1 certificate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase1 import verify_protocol as verify_v1_protocol
from .capability_compiler_phase1_abstention import verify_protocol as verify_v2_protocol
from .capability_compiler_phase1_extract import _sha256_file
from .capability_compiler_phase1_ir import verify_ir


class Phase1CertificateError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Phase1CertificateError(message)


def verify_certificate_data(certificate: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    _require(certificate.get("format") == "abi-capability-compiler-phase1-certificate/1", "unsupported certificate format")
    _require(certificate.get("status") == "PASS", "Phase 1 is not certified")
    for relative, expected in certificate["bindings"].items():
        path = root / relative
        _require(path.is_file() and _sha256_file(path) == expected, f"stale certificate binding: {relative}")
    v1 = verify_v1_protocol(root / "ABI_CAPABILITY_COMPILER_PHASE1_PROTOCOL_V1.json")
    v2 = verify_v2_protocol(root / "ABI_CAPABILITY_COMPILER_PHASE1_ABSTENTION_PROTOCOL_V2.json")
    ir = verify_ir(root / "results/abi_capability_compiler_phase1/final/normalized_acquisition_ir_v1.abicir")
    _require(v1["status"] == v2["status"] == ir["status"] == "PASS", "underlying verifier failed")
    _require(ir["record_count"] == 7000 and ir["records_per_capability"] == 500, "IR depth changed")
    _require(certificate["data_suitability"]["cross_split_near_duplicate_clusters"] == 0, "near-duplicate gate changed")
    _require(certificate["data_suitability"]["specialist_records_in_english_acquisition"] == 0, "English segregation changed")
    _require(certificate["negative_evidence"]["v1_failures_reclassified_by_v2"] == 0, "V1 failures were reclassified")
    _require(certificate["candidate_training_performed"] is False, "training occurred in Phase 1")
    _require(certificate["phase_transition"]["phase1"] == "COMPLETE", "Phase 1 transition changed")
    _require(certificate["phase_transition"]["phase2"] == "OPEN_NOT_STARTED", "Phase 2 status changed")
    return {"status": "PASS", "certificate_id": certificate["certificate_id"], "ir_sha256": ir["archive_sha256"], "phase2": "OPEN_NOT_STARTED", "training_performed": False}


def verify_certificate(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return verify_certificate_data(json.loads(path.read_text(encoding="utf-8")), root=path.parent)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", default="ABI_CAPABILITY_COMPILER_PHASE1_CERTIFICATE_V1.json")
    args = parser.parse_args(argv)
    result = verify_certificate(Path(args.certificate))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
