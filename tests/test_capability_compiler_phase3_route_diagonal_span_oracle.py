from pathlib import Path


def test_route_diagonal_oracle_is_read_only_and_uses_actual_prefix():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_route_diagonal_span_oracle.py"
    ).read_text(encoding="utf-8")
    assert "layer0.forward_with_cache" in text
    assert "layer1.post_attention_norm(attention)" in text
    assert "numerators[route_index]" in text
    assert "feature * diagonal[int(row[\"route\"])]" in text
    assert '"training_performed": False' in text
    assert '"artifact_written": False' in text


def test_route_diagonal_oracle_has_no_parameter_or_artifact_writer():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_route_diagonal_span_oracle.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in text
    assert "save_file" not in text
    assert "_write_immutable(output / \"result.json\"" in text
