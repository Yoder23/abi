from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r89_catalog_is_fresh_seven_digit_and_frozen() -> None:
    path = ROOT / "catalogs/joint_span_validation_r89_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "e21bf7e3a2f3c72d5b82307009fb05c3e1e0623208d9023a13baeaf1e265a118"
    )
    probes = json.loads(path.read_text(encoding="utf-8"))["probes"]
    assert len(probes) == len({row["prompt"] for row in probes}) == 1_400
    assert all("KEY" in row["prompt"] for row in probes)
    assert all(len(row["evaluator"]["value"][-7:]) == 7 for row in probes)
