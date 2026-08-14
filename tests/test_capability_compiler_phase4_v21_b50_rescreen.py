from abi.capability_compiler_phase4_v21_b50_rescreen import (
    FORMAT,
    lexical_guard_contract,
    strong_route_conformance,
)


def _row(**overrides):
    row = {
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


def test_v21_b50_rescreen_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v21-b50-rescreen/1"


def test_strong_route_accepts_exact_output():
    assert strong_route_conformance([_row()], set())


def test_strong_route_accepts_safe_repair_of_historical_failure():
    assert strong_route_conformance(
        [
            _row(
                strong_parent_output_exact=False,
                guard_terminated=True,
                historical_functional_pass_v1=False,
                functional_pass_v1=False,
            )
        ],
        set(),
    )


def test_strong_route_rejects_regression_of_historical_pass():
    assert not strong_route_conformance(
        [
            _row(
                strong_parent_output_exact=False,
                guard_terminated=True,
                historical_functional_pass_v1=True,
                functional_pass_v1=False,
            )
        ],
        set(),
    )


def test_strong_route_rejects_nonprefix_or_collapsed_repair():
    assert not strong_route_conformance(
        [
            _row(
                strong_parent_output_exact=False,
                guard_terminated=True,
                canonical_historical_prefix_preserved=False,
            )
        ],
        set(),
    )


def test_guard_contract_allows_zero_change_zero_collapse_seed():
    assert all(lexical_guard_contract([_row()], set(), "lc-direct-neural-core/21").values())


def test_guard_contract_accepts_safe_historical_collapse_repair():
    repaired = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
        historical_repetition_collapse_v2=True,
    )
    assert all(
        lexical_guard_contract([repaired], set(), "lc-direct-neural-core/21").values()
    )


def test_guard_contract_rejects_change_to_noncollapsed_history():
    changed = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
        historical_repetition_collapse_v2=False,
    )
    gates = lexical_guard_contract([changed], set(), "lc-direct-neural-core/21")
    assert not gates["changed_rows_were_historical_collapses"]
    assert not gates["historical_noncollapsed_outputs_exact"]
    assert not strong_route_conformance(
        [
            _row(
                strong_parent_output_exact=False,
                guard_terminated=True,
                repetition_collapse_v2=True,
            )
        ],
        set(),
    )
