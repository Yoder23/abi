import torch

from abi.capability_compiler_phase3_dual_path_local_fit import (
    _student_components,
    replacement_trainable_keys,
)


class _FakeModel:
    layers = [object()]

    def named_parameters(self):
        return iter(
            (
                ("token_embedding.weight", object()),
                ("layers.0.input_norm.weight", object()),
                ("layers.0.attention_input_projection.weight", object()),
                ("layers.0.attention_norm.weight", object()),
                ("layers.0.attention_output_projection.weight", object()),
                ("layers.0.post_attention_norm.weight", object()),
                ("layers.0.mlp_input_projection.weight", object()),
                ("layers.0.mlp_norm.weight", object()),
                ("layers.0.mlp_output_projection.weight", object()),
                ("final_norm.weight", object()),
            )
        )


def test_dual_path_freezes_only_copied_source_norms() -> None:
    assert replacement_trainable_keys(_FakeModel()) == {
        "layers.0.attention_input_projection.weight",
        "layers.0.attention_norm.weight",
        "layers.0.attention_output_projection.weight",
        "layers.0.mlp_input_projection.weight",
        "layers.0.mlp_norm.weight",
        "layers.0.mlp_output_projection.weight",
    }


class _Identity:
    def __call__(self, value):
        return value


class _ZeroProjection:
    def __call__(self, value):
        return torch.zeros(value.shape[:-1] + (6,), dtype=value.dtype)


class _FakeLayer:
    input_norm = _Identity()
    attention_input_projection = _Identity()
    attention_output_projection = _ZeroProjection()
    o_proj = _Identity()

    def _qkv(self, latent, positions):
        value = latent[:, None]
        return value, value, value

    def _attention(self, query, key, value, *, causal):
        return value

    def _mlp_delta(self, hidden):
        return torch.zeros_like(hidden)


def test_zero_initialized_dual_paths_preserve_residual_stream() -> None:
    hidden = torch.randn(1, 3, 6)
    attention, final = _student_components(_FakeLayer(), hidden, torch.arange(3))
    assert torch.equal(attention, hidden)
    assert torch.equal(final, hidden)
