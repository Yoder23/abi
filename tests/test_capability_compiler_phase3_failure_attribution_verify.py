import json
from pathlib import Path

from abi.capability_compiler_phase3_failure_attribution_verify import verify


ROOT = Path(__file__).resolve().parents[1]


def test_sealed_failure_attribution_evidence_recomputes():
    result = verify(
        ROOT,
        ROOT / "ABI_CAPABILITY_COMPILER_PHASE3_FAILURE_ATTRIBUTION_PROTOCOL_V16.json",
        ROOT / "results/abi_capability_compiler_phase3_failure_attribution/v16/evidence.json",
    )
    assert result["status"] == "PASS"
    assert result["prompts_per_system"] == 280
    assert result["attribution"]["layercake_regression"] is False
    assert result["attribution"]["abi_teacher_payload_signal"] is True
