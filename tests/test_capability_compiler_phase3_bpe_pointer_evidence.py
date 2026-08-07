import hashlib
import json
from pathlib import Path

from abi.capability_compiler_phase2_common import canonical_json_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "results/abi_capability_compiler_phase3_bpe_pointer/development_v54/P0-seed240050"
EVALUATION = ROOT / "results/abi_capability_compiler_phase3_bpe_pointer/evaluation_v54/P0-seed240050"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_v54_pointer_screen_is_bound_and_failed_closed() -> None:
    result = _json(ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_BPE_POINTER_RESULT_V55.json")
    decision = _json(EVALUATION / "decision.json")
    assert result["status"] == "COMPLETE_FAILED_UTF8_BPE_POINTER_SCREEN"
    assert sha256_file(CANDIDATE / "model.safetensors") == result["candidate"]["checkpoint_sha256"]
    assert sha256_file(EVALUATION / "decision.json") == result["evaluation"]["decision_file_sha256"]
    assert sha256_file(EVALUATION / "development_outputs.jsonl") == result["evaluation"]["outputs_sha256"]
    assert decision["functional_passes"] == 785
    assert decision["repetition_collapses"] == 40
    assert decision["route_correct"] == 1400
    assert decision["initial_screen_pass"] is False
    assert decision["promotion_eligible"] is False
    assert decision["phase3_certified"] is False
    assert decision["phase4_open"] is False
    expected = decision.pop("evidence_sha256")
    assert hashlib.sha256(canonical_json_bytes(decision)).hexdigest() == expected
