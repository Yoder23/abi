import json
from pathlib import Path

from abi.capability_compiler_phase2_common import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_PROTOCOL_V1.json"
REPAIR = ROOT / "ABI_CAPABILITY_COMPILER_PHASE2_HUMAN_SCORING_BINDING_REPAIR_V2.json"
REPAIR_SHA256 = "9cd1cacb3224d17c2a6c9a0ee0b4aa4614eff4ee0ff00c036d0437e1cabb8a0c"


def test_phase2_human_scoring_binding_repair_is_exact_and_non_promotional() -> None:
    assert sha256_file(REPAIR) == REPAIR_SHA256
    base = json.loads(BASE.read_text(encoding="utf-8"))
    repair = json.loads(REPAIR.read_text(encoding="utf-8"))
    assert sha256_file(BASE) == repair["base_protocol_sha256"]
    assert repair["completed_preferences_before_repair"] == 0
    assert repair["scientific_protocol_changed"] is False
    assert repair["sealed_packet_changed"] is False
    assert repair["preserved_scoring_contract"] == {
        "validation": base["validation"],
        "scoring": base["scoring"],
        "interpretation": base["interpretation"],
        "custody": base["custody"],
    }
    assert "pyproject.toml" not in repair["effective_implementation_bindings"]
    for relative, expected in repair["effective_implementation_bindings"].items():
        assert sha256_file(ROOT / relative) == expected
    assert "creates no ratings" in repair["claim_boundary"]


def test_phase2_production_packet_remains_blind_and_unrated() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    packet = ROOT / base["sealed_packet"]["path"]
    assert sha256_file(packet) == base["sealed_packet"]["sha256"]
    packet_dir = packet.parent
    rows = 0
    filled = 0
    for index in range(1, 4):
        form = packet_dir / f"rater_form_{index}.jsonl"
        for line in form.read_bytes().splitlines():
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            filled += row["preference"] is not None
    assert rows == 21_000
    assert filled == 0
