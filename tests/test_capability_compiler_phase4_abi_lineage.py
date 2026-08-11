import json
from pathlib import Path

import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_abi_lineage import load_protocol, preflight


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_ABI_LINEAGE_PROTOCOL_V570.json"


def test_all_nested_budgets_preflight_from_clean_start() -> None:
    result = preflight(ROOT, PROTOCOL)
    assert result["status"] == "PASS_PHASE4_ABI_LINEAGE_PREFLIGHT"
    assert result["clean_start_per_budget_and_seed"] is True
    assert result["larger_budget_checkpoint_reuse"] is False
    assert [row["unique_source_attempts"] for row in result["budgets"]] == [1018, 2028, 4005, 7781, 9596]


def test_final_access_mutation_fails_closed(tmp_path: Path) -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["final_test_access"] = "ALLOWED"
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(Phase3Error, match="governance changed"):
        load_protocol(ROOT, path)
