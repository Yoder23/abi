from pathlib import Path


def _text() -> str:
    return (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_factorized_attention_residual_span.py"
    ).read_text(encoding="utf-8")


def test_residual_rank_is_derived_once_behind_locked_factorized_attention():
    text = _text()
    assert "torch.searchsorted(cumulative" in text
    assert '"rank_sweep_performed": False' in text
    assert "locked_rank_schedule" in text
    assert "factorization._rank_schedule" in text


def test_oracle_is_read_only_and_does_not_use_source_mlp_as_prediction():
    text = _text()
    assert "torch.optim" not in text
    assert "save_file" not in text
    assert "source_layer1.mlp(" not in text
    assert '"training_performed": False' in text
    assert '"artifact_written": False' in text
