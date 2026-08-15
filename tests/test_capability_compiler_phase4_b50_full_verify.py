from abi.capability_compiler_phase4_b50_full_verify import (
    FORMAT,
    expected_full_configurations,
)


def test_full_verifier_format_and_matrix_are_frozen():
    assert FORMAT == "abi-capability-compiler-phase4-b50-full-verify/1"
    assert {key: len(value) for key, value in expected_full_configurations().items()} == {
        "L0": 2,
        "L1": 2,
        "D0": 2,
        "D1": 2,
        "D2": 2,
    }


def test_full_matrix_contains_only_derived_configurations():
    matrix = expected_full_configurations()
    assert matrix["L0"] == {(16, 1e-4, 1), (16, 1e-4, 4)}
    assert matrix["L1"] == {(8, 1e-4, 4), (8, 3e-4, 4)}
    for system in ("D0", "D1", "D2"):
        assert matrix[system] == {(None, 3e-5, 2), (None, 3e-5, 4)}
