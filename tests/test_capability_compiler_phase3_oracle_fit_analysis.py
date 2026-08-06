from pathlib import Path

from abi.capability_compiler_phase3_oracle_fit_analysis import analyze


ROOT = Path(__file__).resolve().parents[1]


def test_oracle_fit_raw_failure_recomputes(tmp_path):
    result = analyze(ROOT, ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_ORACLE_FIT_PROTOCOL_V20.json", ROOT / "results/abi_capability_compiler_phase3_oracle_fit/development_v20/O0-seed230003", ROOT / "results/abi_capability_compiler_phase3_oracle_fit/evaluation_v20/O0-seed230003", tmp_path / "decision.json")
    assert result["status"] == "HOST_BRIDGE_EXPRESSIVITY_OR_OPTIMIZATION_LIMITATION"
    assert result["functional_passes"] == 1229
    assert result["repetition_collapses"] == 89
    assert result["promotion_eligible"] is False
