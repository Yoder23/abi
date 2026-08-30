from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from abi_v2.canonical import canonical_json_bytes
from abi_v2.final_validation import evidence_hash
from abi_v2.strict_validation import (
    StrictValidationError,
    verify_reachable_filesystem_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = (
    ROOT
    / "results/abi_final_validation_v2/isolated_certification_strict_r6_full_stream_bound/layercake"
)


def _inputs() -> tuple[dict[str, object], str]:
    isolation = json.loads((EVIDENCE / "physical-isolation.json").read_text(encoding="utf-8"))
    return (
        isolation["reachable_filesystem_forbidden_scan"],
        isolation["mount"]["runtime_site_mount"],
    )


def test_frozen_reachable_inventory_commitment_accepts_exact_raw_rows() -> None:
    summary, runtime_site = _inputs()
    rows = verify_reachable_filesystem_inventory(
        host="layercake",
        path=EVIDENCE / "reachable-filesystem-inventory.jsonl",
        summary=summary,
        runtime_site=runtime_site,
    )
    assert len(rows) == 100_511


def test_rehashed_ordinary_runtime_row_deletion_fails_closed(tmp_path: Path) -> None:
    summary, runtime_site = _inputs()
    altered_summary = copy.deepcopy(summary)
    rows = [
        json.loads(line)
        for line in (EVIDENCE / "reachable-filesystem-inventory.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rows = [row for row in rows if row["path"] != "/usr/bin/NF"]
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    altered = tmp_path / "reachable-filesystem-inventory.jsonl"
    altered.write_bytes(payload)
    altered_summary["inventory_rows"] -= 1
    altered_summary["symlinks_scanned"] -= 1
    altered_summary["inventory_jsonl_sha256"] = hashlib.sha256(payload).hexdigest()
    altered_summary["evidence_sha256"] = evidence_hash(altered_summary)

    with pytest.raises(StrictValidationError, match="release commitment changed"):
        verify_reachable_filesystem_inventory(
            host="layercake",
            path=altered,
            summary=altered_summary,
            runtime_site=runtime_site,
        )
