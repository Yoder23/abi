from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_r86_catalog_is_fresh_complete_and_frozen() -> None:
    path = ROOT / "catalogs/joint_span_validation_r86_v1.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "f16198923847e41e65475a697e531d6b5b319637faa0a3030ae4acbd3c857d78"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    probes = value["probes"]
    assert len(probes) == len({row["prompt"] for row in probes}) == 1_400
    assert all("SUB" in row["prompt"] for row in probes)
    assert all("NODE" not in row["prompt"] for row in probes)
