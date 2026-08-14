from abi.capability_compiler_phase4_contract_baseline_readiness import (
    FORMAT,
    frontier_semantics,
)


EXIT = (
    "smallest passing tested budget and adjacent lower failure reproduced across "
    "three seeds under all fairness views"
)


def test_audit_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-contract-baseline-readiness/1"


def test_frontier_semantics_detects_stricter_later_rule():
    strict = (
        "If all three B40 seeds fail, B50 becomes a minimum candidate. If B40 is "
        "mixed, no stable minimum exists."
    )
    assert all(frontier_semantics(EXIT, EXIT, strict).values())


def test_frontier_semantics_rejects_silent_rule_relaxation():
    changed = EXIT.replace("three seeds", "one seed")
    semantics = frontier_semantics(EXIT, changed, "mixed means no stable minimum")
    assert not semantics["phase0_and_campaign_exit_identical"]
    assert not semantics["later_rule_is_stricter_than_campaign_text"]
