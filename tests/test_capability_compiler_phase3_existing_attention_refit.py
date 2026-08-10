from pathlib import Path


def test_existing_attention_refit_preserves_architecture_and_fail_fast_checkpoint():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_existing_attention_refit.py").read_text(encoding="utf-8")
    assert "ranked[:30] + ranked[32:302]" in text
    assert "attention_prefixes" in text
    assert "if passed:" in text
    assert "source_blocks_in_checkpoint\":0" in text
