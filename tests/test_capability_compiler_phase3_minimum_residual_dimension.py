from pathlib import Path


def test_minimum_residual_dimension_is_derived_once_without_rank_sweep():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_minimum_residual_dimension.py"
    ).read_text(encoding="utf-8")
    assert "torch.searchsorted(cumulative" in text
    assert '"rank_sweep_performed": False' in text
    assert "for rank in" not in text
    assert "layer0.forward_with_cache" in text


def test_minimum_residual_dimension_is_read_only():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_minimum_residual_dimension.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in text
    assert "save_file" not in text
    assert '"training_performed": False' in text
    assert '"artifact_written": False' in text
