from abi.capability_compiler_phase3_progressive_replacement_local_fit_replay import corrected_replacement_trainable_keys


class _Layer:
    pass


class _Fake:
    layers = [_Layer()]

    def named_parameters(self):
        return iter((
            ("layers.0.input_norm.weight", object()),
            ("layers.0.post_attention_norm.weight", object()),
            ("layers.0.latent_input_norm.weight", object()),
            ("layers.0.latent_post_attention_norm.weight", object()),
            ("layers.0.qkv_proj.weight", object()),
        ))


def test_replay_freezes_only_exact_source_norm_names() -> None:
    assert corrected_replacement_trainable_keys(_Fake()) == {
        "layers.0.latent_input_norm.weight",
        "layers.0.latent_post_attention_norm.weight",
        "layers.0.qkv_proj.weight",
    }
