from pathlib import Path


def test_source_functional_diagnostic_has_both_preregistered_arms():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_layer1_source_functional_transplant.py"
    ).read_text(encoding="utf-8")
    assert "layer0.forward_with_cache" in text
    assert '"exact_source_block_on_replacement_prefix"' in text
    assert '"compact_attention_plus_exact_source_mlp"' in text
    assert "dual._teacher_components(teacher, 1, candidate)" in text
    assert "source_layer1.post_attention_layernorm(compact_attention)" in text


def test_source_functional_diagnostic_cannot_promote_source_weights():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_layer1_source_functional_transplant.py"
    ).read_text(encoding="utf-8")
    assert "save_file" not in text
    assert '"source_blocks_promoted": 0' in text
    assert '"training_performed": False' in text
    assert 'protocol.get("source_block_promotion") != "PROHIBITED"' in text
