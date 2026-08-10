from abi.capability_compiler_phase3_progressive_replacement_verify import expected_keys


def test_progressive_replacement_verifier_requires_every_copied_tensor() -> None:
    keys = expected_keys(32)
    assert len(keys) == 67
    assert {"token_embedding.weight", "lm_head.weight", "final_norm.weight"} <= keys
    assert "layers.0.input_norm.weight" in keys
    assert "layers.31.post_attention_norm.weight" in keys
