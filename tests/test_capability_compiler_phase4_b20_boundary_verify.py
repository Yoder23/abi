from abi.capability_compiler_phase4_b20_boundary_verify import (
    _oracle_reconstruction,
)


def test_oracle_reconstruction_requires_a_surviving_failure():
    rows = [
        {
            "probe_id": str(index),
            "capability": "fluent_realization",
            "functional_pass_v1": index < 90,
            "functional_pass_v2": index < 90,
            "repetition_collapse_v2": False,
        }
        for index in range(100)
    ]
    result = _oracle_reconstruction(
        rows,
        {"functional_passes_v1": 90, "repetition_collapses_v2": 0},
        {
            "per_capability_point": 0.9,
            "per_capability_lower": 0.85,
            "critical_point": 0.95,
            "critical_lower": 0.9,
        },
    )
    assert result["oracle_gate_failures"]["fluent_realization"]
    assert not result["failure_proven"]  # The production verifier also requires 1,400 rows.
