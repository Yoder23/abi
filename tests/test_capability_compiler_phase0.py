import copy
import json
from pathlib import Path

from abi.capability_compiler_phase0 import REQUIRED_SYSTEMS, validate_protocol


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "ABI_CAPABILITY_COMPILER_PHASE0_PROTOCOL_V1.json"


def load_protocol():
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_checked_in_phase0_protocol_passes_fail_closed_verifier():
    assert validate_protocol(load_protocol()) == []


def test_phase0_protocol_has_exact_mandatory_systems_and_bounded_search():
    protocol = load_protocol()
    assert set(protocol["mandatory_systems"]) == REQUIRED_SYSTEMS
    for system_id in ("L0", "L1", "D0", "D1", "D2"):
        assert 1 <= protocol["mandatory_systems"][system_id]["maximum_development_configurations"] <= 8


def test_phase0_protocol_rejects_final_selection_and_weak_statistics():
    protocol = load_protocol()
    protocol["data_boundaries"]["final_data_may_select"] = ["checkpoint"]
    protocol["statistics"]["headline_runtime_repetitions_minimum"] = 2
    errors = validate_protocol(protocol)
    assert "final data cannot select anything" in errors
    assert "headline runtime requires at least 20 repetitions" in errors


def test_phase0_protocol_rejects_missing_baseline_and_teacher_at_inference():
    protocol = load_protocol()
    del protocol["mandatory_systems"]["L1"]
    protocol["mandatory_systems"]["A0"]["teacher_absent_at_inference"] = False
    errors = validate_protocol(protocol)
    assert "mandatory comparison system set is incomplete or changed" in errors
    assert "A0 must remove the teacher at inference" in errors


def test_phase0_protocol_rejects_leakage_sparse_and_speed_relaxation():
    protocol = load_protocol()
    protocol["segregation_and_exclusion_gates"]["specialist_records_in_english_artifact_maximum"] = 1
    protocol["segregation_and_exclusion_gates"]["inactive_capability_execution_events_maximum"] = 1
    protocol["integrated_layercake_gates"]["cpu_throughput_ratio_vs_optimized_transformer_minimum"] = 1.99
    errors = validate_protocol(protocol)
    assert "specialist_records_in_english_artifact_maximum must be zero" in errors
    assert "inactive_capability_execution_events_maximum must be zero" in errors
    assert "LayerCake CPU throughput ratio gate must be at least 2x" in errors


def test_phase0_protocol_rejects_null_and_universal_superiority():
    protocol = copy.deepcopy(load_protocol())
    protocol["product_superiority_gates"]["universal_superiority_claim_allowed"] = True
    protocol["optional_teacher_improvement_gate"]["protected_dimension_noninferiority_margin"] = None
    errors = validate_protocol(protocol)
    assert "universal superiority claim must be prohibited" in errors
    assert "protocol cannot contain null values" in errors


def test_current_docs_name_phase0_and_do_not_claim_superiority():
    status = (ROOT / "CURRENT_PROJECT_STATUS.md").read_text(encoding="utf-8")
    claims = (ROOT / "CLAIMS.md").read_text(encoding="utf-8")
    mission = (ROOT / "ACTIVE_MISSION.md").read_text(encoding="utf-8")
    assert "Phase 0" in status
    assert "has not established superiority" in claims
    assert "Phase 0" in mission
