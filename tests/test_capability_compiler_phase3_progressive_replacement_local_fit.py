from abi.capability_compiler_phase3_progressive_replacement_local_fit import replacement_trainable_keys


class _Fake:
    def named_parameters(self):
        return iter((
            ("token_embedding.weight", object()),
            ("layers.0.input_norm.weight", object()),
            ("layers.0.post_attention_norm.weight", object()),
            ("layers.0.input_projection.weight", object()),
            ("layers.0.qkv_proj.weight", object()),
            ("layers.0.output_projection.weight", object()),
            ("final_norm.weight", object()),
            ("lm_head.weight", object()),
        ))


def test_local_fit_freezes_copied_substrate_and_trains_only_replacement_values() -> None:
    assert replacement_trainable_keys(_Fake()) == {
        "layers.0.input_projection.weight",
        "layers.0.qkv_proj.weight",
        "layers.0.output_projection.weight",
    }
