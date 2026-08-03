from pathlib import Path

from abi.capability_compiler_phase1_abstention import verify_protocol


ROOT = Path(__file__).resolve().parents[1]


def test_v2_abstention_protocol_is_fresh_and_bounded():
    result = verify_protocol(ROOT / "ABI_CAPABILITY_COMPILER_PHASE1_ABSTENTION_PROTOCOL_V2.json")
    assert result["status"] == "PASS"
    assert result["fresh_prompts"] == 400
    assert result["minimum_fresh_passes"] == 263
    assert result["training_authorized"] is False
