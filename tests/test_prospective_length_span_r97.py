from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.foreign_capability_r14.core import evidence_hash
from experiments.prospective_length_span_r97.verify_evidence_v1 import verify

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "prospective_length_span_r97"


def _read_verified_receipt(name: str) -> dict:
    path = RESULTS / name
    value = json.loads(path.read_text(encoding="utf-8"))
    claim = value.get("evidence_sha256")
    unsigned = dict(value)
    unsigned.pop("evidence_sha256", None)
    assert evidence_hash(unsigned) == claim
    return value


def test_r97_recomputes_from_all_bound_raw_rows() -> None:
    value = verify(ROOT)
    assert value == {
        "rows": 1400,
        "candidate_passing": 1399,
        "source_passing": 1088,
        "parent_passing": 60,
        "random_passing": 17,
        "source_retained": 1087,
        "candidate_collapses": 0,
        "sparse_rows": 1400,
        "raw_sha256": (
            "b4a4999228af61a509a7d22f8f6906b3075ce649a52e531b959646fbbdb40b8c"
        ),
        "result_sha256": (
            "e50c92e0da7f70ddcbc4b93a1e716b0c5fdf0ca1cdab8b45b5e10c1b7eb34f14"
        ),
    }


def test_r97_hostile_and_live_replay_receipts_are_content_bound() -> None:
    hostile = _read_verified_receipt("hostile_verification_v1.json")
    assert hostile["verdict"] == "PASS_R97_HOSTILE_AUDIT"
    assert hostile["hostile_cases"] == hostile["rejected_cases"] == 12
    assert hostile["accepted_cases"] == 0
    assert {row["verdict"] for row in hostile["observations"]} == {"REJECTED"}

    replay = _read_verified_receipt("live_replay_receipt_v1.json")
    raw_path = RESULTS / "live_replay_v1" / "evaluation.jsonl"
    raw_sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert replay["verdict"] == "PASS_R97_LIVE_REPLAY"
    assert replay["rows"] == 1400
    assert replay["raw_rows_byte_identical"] is True
    assert raw_sha256 == replay["primary_raw_sha256"]
    assert raw_sha256 == replay["replay_raw_sha256"]
    assert replay["full_abi_moonshot"] == "OPEN"
