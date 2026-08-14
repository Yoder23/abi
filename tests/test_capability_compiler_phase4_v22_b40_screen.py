from abi.capability_compiler_phase4_v22_b40_screen import FORMAT, b40_preservation


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


def test_v22_b40_screen_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v22-b40-screen/1"


def test_b40_preservation_accepts_exact_and_safe_collapse_repair():
    repaired = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
        historical_repetition_collapse_v2=True,
    )
    assert all(b40_preservation([_row(), repaired], set(), "lc-direct-neural-core/22").values())


def test_b40_preservation_allows_format_change_through_strong_contract():
    changed = _row(
        capability="format_control",
        strong_parent_output_exact=False,
        output_changed_from_v19_history=True,
        v22_format={"mode": "deterministic_prompt_literal_transducer"},
    )
    assert all(b40_preservation([changed], set(), "lc-direct-neural-core/22").values())


def test_b40_preservation_rejects_unexplained_nonformat_change():
    changed = _row(
        strong_parent_output_exact=False,
        guard_terminated=True,
        output_changed_from_v19_history=True,
    )
    gates = b40_preservation([changed], set(), "lc-direct-neural-core/22")
    assert not gates["nonformat_change_set_equals_historical_collapses"]
    assert not gates["all_other_nonformat_outputs_exact"]


def test_b40_preservation_rejects_unrepaired_collapse_or_wrong_interface():
    collapse = _row(historical_repetition_collapse_v2=True)
    gates = b40_preservation([collapse], set(), "lc-direct-neural-core/22")
    assert not gates["nonformat_change_set_equals_historical_collapses"]
    assert not gates["all_other_nonformat_outputs_exact"]
    assert not b40_preservation([_row()], set(), "lc-direct-neural-core/21")[
        "interface_v22_declared"
    ]
