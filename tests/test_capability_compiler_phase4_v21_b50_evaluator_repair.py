from abi.capability_compiler_phase4_v21_b50_evaluator_repair import (
    FORMAT,
    corrected_guard_contract,
)


def _row(**overrides):
    row = {
        "probe_id": "p1",
        "capability": "grammar",
        "strong_parent_output_exact": True,
        "guard_terminated": False,
        "canonical_historical_prefix_preserved": True,
        "historical_functional_pass_v1": True,
        "functional_pass_v1": True,
        "repetition_collapse_v2": False,
        "output_changed_from_v19_history": False,
        "historical_repetition_collapse_v2": False,
    }
    row.update(overrides)
    return row


def test_repair_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v21-b50-evaluator-repair/1"


def test_coherence_change_is_governed_outside_noncoherence_identity():
    coherence = _row(
        capability="coherence",
        strong_parent_output_exact=False,
        output_changed_from_v19_history=True,
    )
    gates = corrected_guard_contract(
        [_row(probe_id="strong-control"), coherence],
        {"coherence"},
        "lc-direct-neural-core/21",
    )
    assert all(gates.values())


def test_exact_historical_collapse_repair_passes():
    repaired = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
        historical_repetition_collapse_v2=True,
    )
    assert all(
        corrected_guard_contract([repaired], set(), "lc-direct-neural-core/21").values()
    )


def test_unexpected_noncoherence_change_fails():
    unexpected = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
    )
    gates = corrected_guard_contract([unexpected], set(), "lc-direct-neural-core/21")
    assert not gates["noncoherence_change_set_equals_historical_collapses"]
    assert not gates["all_other_noncoherence_outputs_exact"]


def test_unrepaired_historical_collapse_fails():
    unchanged_collapse = _row(historical_repetition_collapse_v2=True)
    gates = corrected_guard_contract(
        [unchanged_collapse], set(), "lc-direct-neural-core/21"
    )
    assert not gates["noncoherence_change_set_equals_historical_collapses"]
    assert not gates["all_other_noncoherence_outputs_exact"]


def test_guard_or_prefix_or_functional_regression_fails():
    base = {
        "strong_parent_output_exact": False,
        "guard_terminated": True,
        "output_changed_from_v19_history": True,
        "historical_repetition_collapse_v2": True,
    }
    assert not corrected_guard_contract(
        [_row(**{**base, "guard_terminated": False})],
        set(),
        "lc-direct-neural-core/21",
    )["changed_rows_guard_terminated"]
    assert not corrected_guard_contract(
        [_row(**base, canonical_historical_prefix_preserved=False)],
        set(),
        "lc-direct-neural-core/21",
    )["changed_rows_canonical_prefixes"]
    assert not corrected_guard_contract(
        [_row(**base, functional_pass_v1=False)], set(), "lc-direct-neural-core/21"
    )["historical_functional_passes_preserved"]
