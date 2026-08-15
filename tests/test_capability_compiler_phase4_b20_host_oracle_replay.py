from abi.capability_compiler_phase4_b20_host_oracle_replay import (
    _gate_failures,
    capability_metrics,
    oracle_rows,
)


def _row(capability: str, passed: bool, collapse: bool = False, probe: int = 0):
    return {
        "capability": capability,
        "functional_pass_v1": passed,
        "functional_pass_v2": passed,
        "repetition_collapse_v2": collapse,
        "probe_id": f"p{probe}",
    }


def test_oracle_grants_only_declared_host_scope():
    rows = [
        _row("clarification", False, probe=1),
        _row("fluent_realization", False, probe=2),
        _row("rewriting", False, collapse=True, probe=3),
    ]
    replayed = oracle_rows(rows)
    assert replayed[0]["functional_pass_v1"] is True
    assert replayed[1] == rows[1]
    assert replayed[2]["functional_pass_v1"] is True
    assert replayed[2]["repetition_collapse_v2"] is False


def test_immutable_quality_failure_survives_oracle():
    rows = [_row("fluent_realization", index < 91, probe=index) for index in range(100)]
    metrics = capability_metrics(rows)
    failures = _gate_failures(
        metrics,
        {
            "per_capability_point": 0.9,
            "per_capability_lower": 0.85,
            "critical_point": 0.95,
            "critical_lower": 0.9,
        },
    )
    assert failures["fluent_realization"]["wilson"]["lower_95"] < 0.85
