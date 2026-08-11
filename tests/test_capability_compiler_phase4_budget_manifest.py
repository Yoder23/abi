import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_budget_manifest import load_protocol, run


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_BUDGET_MANIFEST_PROTOCOL_V566.json"


def test_real_budget_manifest_is_nested_and_closes_full_consumption() -> None:
    result = run(ROOT, PROTOCOL)
    assert result["status"] == "PASS_NESTED_BUDGET_MANIFEST_TRAINING_PROTOCOL_MAY_BE_SEALED"
    assert [row["fraction"] for row in result["budgets"]] == [0.1, 0.2, 0.4, 0.8, 1.0]
    assert result["budgets"][-1]["unique_source_attempts"] == 9596
    assert result["budgets"][-1]["authoritative_teacher_output_tokens"] == 294212


def test_training_authorization_is_rejected(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["neural_training_authorized"] = True
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(ROOT, path)
