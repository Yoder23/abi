from pathlib import Path


def _text():
    return (Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_source_nonlinear_coefficient_oracle.py").read_text(encoding="utf-8")


def test_source_nonlinear_is_projected_through_fixed_basis():
    text=_text()
    assert "source_layer1.mlp(" in text
    assert "rank_audit.project_with_basis(source_delta" in text
    assert '"residual_rank":rank' in text


def test_oracle_writes_no_model_and_performs_no_fit():
    text=_text()
    assert "save_file" not in text
    assert "torch.optim" not in text
    assert '"training_performed":False' in text
    assert '"artifact_written":False' in text
