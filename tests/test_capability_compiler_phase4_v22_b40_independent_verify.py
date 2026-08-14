from abi.capability_compiler_phase4_v22_b40_independent_verify import (
    FORMAT,
    independent_b40_preservation,
)


def _row(**overrides):
    row = {
        "probe_id": "p1",
        "capability": "grammar",
        "strong_parent_output_exact": True,
        "v22_format": {},
        "functional_pass_v1": True,
        "historical_functional_pass_v1": True,
        "repetition_collapse_v2": False,
        "historical_repetition_collapse_v2": False,
        "guard_terminated": False,
        "strong_parent_prefix_preserved": True,
        "canonical_historical_prefix_preserved": True,
        "output_changed_from_v19_history": False,
    }
    row.update(overrides)
    return row


def test_verifier_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v22-b40-independent-verify/1"


def test_independent_preservation_accepts_exact_and_collapse_repair():
    repaired = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        historical_repetition_collapse_v2=True,
        output_changed_from_v19_history=True,
    )
    assert all(
        independent_b40_preservation(
            [_row(), repaired], set(), "lc-direct-neural-core/22"
        ).values()
    )


def test_independent_preservation_rejects_unexplained_drift():
    changed = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
    )
    gates = independent_b40_preservation(
        [changed], set(), "lc-direct-neural-core/22"
    )
    assert not gates["nonformat_change_set_equals_historical_collapses"]
    assert not gates["all_other_nonformat_outputs_exact"]


def test_independent_preservation_rejects_bad_guard_repair():
    repaired = _row(
        strong_parent_output_exact=False,
        historical_repetition_collapse_v2=True,
        output_changed_from_v19_history=True,
        guard_terminated=False,
    )
    gates = independent_b40_preservation(
        [repaired], set(), "lc-direct-neural-core/22"
    )
    assert not gates["changed_nonformat_guard_terminated"]
