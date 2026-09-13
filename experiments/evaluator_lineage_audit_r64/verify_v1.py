"""Fail-closed recomputation verifier for R64."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from experiments.evaluator_lineage_audit_r64.audit_v1 import run, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    expected_result = evidence / "result.json"
    expected_rows = evidence / "row_audit.jsonl"
    if not expected_result.is_file() or not expected_rows.is_file():
        raise SystemExit("FAIL: R64 evidence is missing")
    with tempfile.TemporaryDirectory(prefix="abi-r64-verify-") as directory:
        regenerated = Path(directory)
        run(args.root.resolve(), regenerated)
        if (
            sha256_file(regenerated / "result.json") != sha256_file(expected_result)
            or sha256_file(regenerated / "row_audit.jsonl") != sha256_file(expected_rows)
        ):
            raise SystemExit("FAIL: R64 evidence is not exactly recomputable")
    result = json.loads(expected_result.read_text(encoding="utf-8"))
    if result.get("promotion_eligible") is not False or result.get("full_abi_moonshot") != "OPEN":
        raise SystemExit("FAIL: R64 overclaims its post-hoc audit")
    print("PASS: R64 evidence recomputes exactly and remains non-promotional")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
