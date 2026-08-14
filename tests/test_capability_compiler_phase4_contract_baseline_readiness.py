from abi.capability_compiler_phase4_contract_baseline_readiness import (
    FORMAT,
    frontier_semantics,
    phase2_original_information,
)
from abi.capability_compiler_phase3 import Phase3Error
import pytest


EXIT = (
    "smallest passing tested budget and adjacent lower failure reproduced across "
    "three seeds under all fairness views"
)


def test_audit_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-contract-baseline-readiness/3"


def test_frontier_semantics_detects_stricter_later_rule():
    strict = (
        "If all three B40 seeds fail, B50 becomes a minimum candidate. If B40 is "
        "mixed, no stable minimum exists."
    )
    frozen = {
        "mandatory_system_specs_frozen": True,
        "data_boundaries_frozen": True,
        "numeric_gates_frozen": True,
        "statistics_frozen": True,
        "accounting_frozen": True,
        "stop_rules_frozen": True,
    }
    assert all(frontier_semantics(EXIT, strict, frozen).values())


def test_frontier_semantics_rejects_silent_rule_relaxation():
    changed = EXIT.replace("three seeds", "one seed")
    frozen = {
        "mandatory_system_specs_frozen": True,
        "data_boundaries_frozen": True,
        "numeric_gates_frozen": True,
        "statistics_frozen": True,
        "accounting_frozen": True,
        "stop_rules_frozen": False,
    }
    semantics = frontier_semantics(changed, "mixed means no stable minimum", frozen)
    assert not semantics["phase0_freezes_mandatory_specs"]
    assert not semantics["campaign_requires_three_seeds"]
    assert not semantics["later_rule_is_stricter_than_campaign_text"]


def test_phase2_wrapper_must_link_the_bound_original_protocol():
    repair = {"original_protocol": {"path": "original.json", "sha256": "abc"}}
    original = {
        "phase1": {
            "selected_teacher_input_tokens": 10,
            "selected_teacher_output_tokens": 20,
            "selected_raw_prompt_bytes": 30,
            "selected_raw_teacher_output_bytes": 40,
        }
    }
    assert phase2_original_information(repair, original, "original.json", "abc") == {
        "selected_teacher_input_tokens": 10,
        "selected_teacher_output_tokens": 20,
        "selected_raw_prompt_bytes": 30,
        "selected_raw_teacher_output_bytes": 40,
    }
    with pytest.raises(Phase3Error):
        phase2_original_information(repair, original, "changed.json", "abc")
