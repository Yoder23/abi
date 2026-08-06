from pathlib import Path

from abi.capability_compiler_phase3_self_prefix_analysis import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_v17_raw_decision_recomputes(tmp_path):
    result = analyze(ROOT, ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_SELF_PREFIX_PROTOCOL_V17.json", ROOT / "results/abi_capability_compiler_phase3_self_prefix", tmp_path / "decision.json")
    assert result["status"] == "FAIL_INITIAL_SEED_SELF_PREFIX_SUCCESSOR"
    assert result["systems"]["S0"]["functional_passes"] == 1185
    assert result["systems"]["S1"]["functional_passes"] == 1208
    assert result["gates"]["S0_beats_compute_matched_S1"] is False
    assert result["decision"]["remaining_two_seeds_authorized"] is False
