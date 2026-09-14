from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r87_catalog_is_fresh_complete_and_frozen() -> None:
    path = ROOT / "catalogs/joint_span_validation_r87_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "d956c7ab2eb6dc4e6d099cdaf79c416751c0bf1b3601da181fbbf2ef05e07926"
    )
    probes = json.loads(path.read_text(encoding="utf-8"))["probes"]
    assert len(probes) == len({row["prompt"] for row in probes}) == 1_400
    assert all("ENT" in row["prompt"] for row in probes)
    assert all("NODE" not in row["prompt"] and "OBJ" not in row["prompt"] for row in probes)
