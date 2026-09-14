from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r88_catalog_is_fresh_complete_and_frozen() -> None:
    path = ROOT / "catalogs/joint_span_validation_r88_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "5dba762eec849f2e6531f5f1283417d38e9470a9b37ca69cf3576bb25286a177"
    )
    probes = json.loads(path.read_text(encoding="utf-8"))["probes"]
    assert len(probes) == len({row["prompt"] for row in probes}) == 1_400
    assert all("REF" in row["prompt"] for row in probes)
    assert all("NODE" not in row["prompt"] and "OBJ" not in row["prompt"] for row in probes)
