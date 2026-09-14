from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r85_catalog_is_fresh_complete_and_frozen() -> None:
    path = ROOT / "catalogs/stateful_span_validation_r85_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "8ad899dfb15f120ae3ffdbde6b6d8dcb06bbb265883665bcecdeff048eaafc4c"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    probes = value["probes"]
    assert len(probes) == len({row["prompt"] for row in probes}) == 1_400
    assert all("NODE" in row["prompt"] and "VERTEX" not in row["prompt"] for row in probes)
