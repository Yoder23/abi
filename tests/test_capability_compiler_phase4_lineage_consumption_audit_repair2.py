from pathlib import Path

from abi.capability_compiler_phase4_lineage_consumption_audit_repair2 import run


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "ABI_CAPABILITY_COMPILER_PHASE4_LINEAGE_CONSUMPTION_AUDIT_REPAIR2_PROTOCOL_V564.json"


def test_all_negative_diagnostic_polarities_are_positive_assertions() -> None:
    result = run(ROOT, PROTOCOL)
    assert result["status"] == "PASS_CONSUMED_INFORMATION_LINEAGE_FRONTIER_PROTOCOL_OPEN"
    assert all(value is True for value in result["gates"].values())
    assert result["implementation_repair"]["scientific_fields_changed"] is False
    assert result["consumed_unique_information"]["source_attempts"] == 9596
    assert result["consumed_unique_information"]["authoritative_teacher_output_tokens"] == 294212
