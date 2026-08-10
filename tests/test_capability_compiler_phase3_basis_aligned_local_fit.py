from abi.capability_compiler_phase3_basis_aligned_local_fit import key_classes


class _Fake:
    layers = [object()]

    def named_parameters(self):
        return iter((
            ("layers.0.input_norm.weight", object()),
            ("layers.0.attention_input_projection.weight", object()),
            ("layers.0.post_attention_norm.weight", object()),
            ("layers.0.mlp_input_projection.weight", object()),
            ("layers.0.mlp_output_projection.weight", object()),
            ("layers.0.mlp_residual_mean", object()),
        ))


def test_basis_mean_are_imported_and_coefficient_map_is_trainable() -> None:
    trainable, imported = key_classes(_Fake())
    assert imported == {"layers.0.mlp_output_projection.weight", "layers.0.mlp_residual_mean"}
    assert trainable == {"layers.0.attention_input_projection.weight", "layers.0.mlp_input_projection.weight"}
