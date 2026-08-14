from abi.capability_compiler_phase4_v22_b50_independent_verify import (
    FORMAT,
    format_semantics,
    independent_strong_route_conformance,
)


def _format_row(**overrides):
    row = {
        "output": "alpha\nbeta",
        "v22_format": {
            "mode": "literal-mode",
            "deterministic_transducer": True,
            "prompt_prefill_forward_passes": 1,
            "candidate_scoring_forward_passes": 0,
            "decode_forward_passes": 0,
            "persistent_prompt_state_created": True,
            "model_state_advanced_after_prefill": False,
            "active_residual_routes": 0,
            "evaluator_used": False,
            "teacher_used": False,
        },
    }
    row.update(overrides)
    return row


def _strong_row(**overrides):
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


def test_verifier_has_separate_versioned_format():
    assert FORMAT == "abi-capability-compiler-phase4-v22-b50-independent-verify/1"


def test_format_semantics_accepts_only_exact_declared_execution():
    checks = format_semantics(
        _format_row(),
        "prompt",
        lambda _: ("alpha", "beta"),
        lambda values: "\n".join(values),
        "literal-mode",
    )
    assert all(checks.values())


def test_format_semantics_rejects_output_or_execution_mutation():
    wrong_output = format_semantics(
        _format_row(output="alpha beta"),
        "prompt",
        lambda _: ("alpha", "beta"),
        lambda values: "\n".join(values),
        "literal-mode",
    )
    assert not wrong_output["exact_render"]
    wrong_execution = _format_row(
        v22_format={**_format_row()["v22_format"], "decode_forward_passes": 1}
    )
    checks = format_semantics(
        wrong_execution,
        "prompt",
        lambda _: ("alpha", "beta"),
        lambda values: "\n".join(values),
        "literal-mode",
    )
    assert not checks["zero_decode"]


def test_independent_strong_contract_accepts_exact_format_and_safe_guard():
    assert independent_strong_route_conformance([_strong_row()], set())
    assert independent_strong_route_conformance(
        [
            _strong_row(
                capability="format_control",
                strong_parent_output_exact=False,
                v22_format={"mode": "literal-mode"},
            )
        ],
        set(),
    )
    assert independent_strong_route_conformance(
        [
            _strong_row(
                strong_parent_output_exact=False,
                guard_terminated=True,
                historical_functional_pass_v1=False,
                functional_pass_v1=False,
            )
        ],
        set(),
    )


def test_independent_strong_contract_rejects_unlabeled_format_change():
    assert not independent_strong_route_conformance(
        [
            _strong_row(
                capability="format_control",
                strong_parent_output_exact=False,
                v22_format={},
            )
        ],
        set(),
    )
