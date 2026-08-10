from pathlib import Path


def test_attention_span_oracle_uses_rank384_and_actual_prefix():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_layer1_attention_span_oracle.py"
    ).read_text(encoding="utf-8")
    assert 'int(protocol.get("rank", 0)) != 384' in text
    assert "layer0.forward_with_cache" in text
    assert "exact_attention, _ = dual._teacher_components(teacher, 1, candidate)" in text
    assert "reconstructed_delta = mean + ((delta - mean) @ basis)" in text


def test_attention_span_oracle_is_read_only_and_source_diagnostic_only():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_layer1_attention_span_oracle.py"
    ).read_text(encoding="utf-8")
    assert "save_file" not in text
    assert "torch.optim" not in text
    assert '"source_blocks_promoted": 0' in text
    assert '"training_performed": False' in text
