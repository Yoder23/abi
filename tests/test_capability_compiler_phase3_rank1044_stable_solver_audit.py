from pathlib import Path


def test_stable_solver_keeps_equations_and_changes_only_numeric_execution():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_rank1044_stable_solver_audit.py"
    ).read_text(encoding="utf-8")
    assert "gram.add_((feature_chunk.transpose(0, 1) @ feature_chunk).double())" in text
    assert "cross.add_((feature_chunk.transpose(0, 1) @ target_chunk).double())" in text
    assert "torch.linalg.solve(system, cross).float()" in text
    assert "original.closed.solve_ridge = replacement" in text
    assert "original.execute(root, protocol_path" in text


def test_stable_solver_checks_optimality_and_preserves_raw_replay():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_rank1044_stable_solver_audit.py"
    ).read_text(encoding="utf-8")
    assert '"penalized_objective_ratio_to_zero"' in text
    assert '"sequential_training_bound_valid"' in text
    assert 'output / "solver_replay"' in text
    assert "torch.optim" not in text
    assert "save_file" not in text
