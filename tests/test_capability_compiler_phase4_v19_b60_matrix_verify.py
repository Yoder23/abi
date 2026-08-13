from abi.capability_compiler_phase4_v19_b60_matrix_verify import matrix_decision


def test_mixed_matrix_stops_without_authorizing_adjacent_budgets():
    rows = [
        {"seed":104729,"machine_gates_pass":False},
        {"seed":130363,"machine_gates_pass":True},
        {"seed":155921,"machine_gates_pass":True},
    ]
    result = matrix_decision(rows)
    assert result["exact_registered_seeds"]
    assert result["mixed"] and result["stop_refinement"]
    assert not result["b50_authorized"] and not result["b70_authorized"]
    assert not result["stable_minimum_proven"]


def test_all_pass_would_authorize_only_b50():
    rows = [{"seed":seed,"machine_gates_pass":True} for seed in (104729,130363,155921)]
    result = matrix_decision(rows)
    assert result["all_seed_pass"] and result["b50_authorized"]
    assert not result["mixed"] and not result["b70_authorized"]
