from pathlib import Path


def _text() -> str:
    return (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_factorized_layer1_analytic_realization.py"
    ).read_text(encoding="utf-8")


def test_realization_uses_fixed_factorization_and_stable_analytic_solver():
    text = _text()
    assert "span._factors(" in text
    assert "stable._stable_solver_factory(" in text
    assert "locked_rank_schedule" in text
    assert "torch.optim" not in text


def test_component_is_pass_only_and_contains_no_complete_source_block():
    text = _text()
    assert "if passed:" in text
    assert '"attention_q.weight"' in text
    assert '"attention_value_factors.weight"' in text
    assert '"route_coefficient.weight"' in text
    assert "source_layer1.state_dict" not in text
    assert '"source_blocks_promoted": 0' in text
