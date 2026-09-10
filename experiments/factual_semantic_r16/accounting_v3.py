"""Bind corrected R16 accounting to the exact run receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import evidence_hash, sha256_file, write_json_once

from .accounting_v2 import account_v2


def account_v3(run_dir: Path) -> dict[str, Any]:
    base = account_v2(run_dir)
    payload = {key: value for key, value in base.items() if key != "evidence_sha256"}
    if payload.get("repair_of") is None:
        payload["legacy_recomputation_evidence_sha256"] = payload.pop(
            "superseded_v1_evidence_sha256"
        )
    result = {
        **payload,
        "format": "abi-r16-imported-information-accounting/3",
        "base_accounting_evidence_sha256": base["evidence_sha256"],
        "verified_identity": {
            "run_receipt_sha256": sha256_file(run_dir / "receipt.json"),
        },
    }
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = account_v3(args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
