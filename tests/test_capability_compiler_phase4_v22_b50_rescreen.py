from abi.capability_compiler_phase4_v22_b50_rescreen import (
    FORMAT,
    strong_route_conformance,
)


def _row(**overrides):
    row = {
        "capability": "grammar",
        "strong_parent_output_exact": True,
        "v22_format": {},
        "functional_pass_v1": True,
        "historical_functional_pass_v1": True,
        "repetition_collapse_v2": False,
        "guard_terminated": False,
        "strong_parent_prefix_preserved": True,
    }
    row.update(overrides)
    return row


def test_v22_b50_rescreen_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v22-b50-rescreen/1"


def test_strong_route_accepts_exact_and_explicit_format_paths():
    assert strong_route_conformance([_row()], set())
    assert strong_route_conformance(
        [
            _row(
                capability="format_control",
                strong_parent_output_exact=False,
                v22_format={"mode": "deterministic_prompt_literal_transducer"},
            )
        ],
        set(),
    )


def test_strong_route_rejects_unlabeled_or_failed_format_change():
    assert not strong_route_conformance(
        [_row(capability="format_control", strong_parent_output_exact=False)], set()
    )
    assert not strong_route_conformance(
        [
            _row(
                capability="format_control",
                strong_parent_output_exact=False,
                v22_format={"mode": "deterministic_prompt_literal_transducer"},
                functional_pass_v1=False,
            )
        ],
        set(),
    )


def test_strong_route_retains_safe_lexical_guard_repair():
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
