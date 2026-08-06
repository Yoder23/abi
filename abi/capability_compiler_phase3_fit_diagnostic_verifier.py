"""Independent verifier for the sealed V26/V27 read-only diagnostic."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .capability_compiler_phase2_common import canonical_json_bytes, sha256_file
from .capability_compiler_phase3 import Phase3Error
from .capability_compiler_phase3_direct_core import _json
from .capability_compiler_phase3_fit_diagnostic import execute


RESULT_FORMAT = "abi-capability-compiler-phase3-fit-diagnostic-result/1"


def embedded_evidence_sha256(document: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(document))
    claimed = payload.pop("evidence_sha256", None)
    if not isinstance(claimed, str):
        raise Phase3Error("V26 evidence hash is missing")
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_document(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    if stored.get("format") != RESULT_FORMAT or stored.get("status") != "PASS_READ_ONLY_ATTRIBUTION":
        raise Phase3Error("V26 result identity changed")
    if stored.get("phase3_certified") is not False or stored.get("final_test_accessed") is not False:
        raise Phase3Error("V26 governance changed")
    if embedded_evidence_sha256(stored) != stored.get("evidence_sha256"):
        raise Phase3Error("V26 embedded evidence hash changed")
    if dict(stored) != dict(expected):
        raise Phase3Error("V26 result differs from independent GPU recomputation")
    return {
        "status": "PASS",
        "evidence_sha256": str(stored["evidence_sha256"]),
        "phase3_certified": False,
        "superiority_established": False,
    }


def verify(root: Path, protocol_path: Path, result_path: Path) -> dict[str, Any]:
    if not result_path.is_file():
        raise Phase3Error("V26 result is missing")
    stored = _json(result_path)
    expected = execute(root, protocol_path)
    return {
        **verify_document(stored, expected),
        "result_file_sha256": sha256_file(result_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="ABI_CAPABILITY_COMPILER_PHASE3_FIT_DIAGNOSTIC_RUNTIME_REPAIR1_V27.json")
    parser.add_argument("--result", default="results/abi_capability_compiler_phase3_fit_diagnostic/fit_generalization_v26.json")
    args = parser.parse_args(argv)
    root = Path.cwd().resolve()
    result = verify(root, (root / args.protocol).resolve(), (root / args.result).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
