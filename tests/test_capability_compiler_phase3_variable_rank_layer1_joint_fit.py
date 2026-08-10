from pathlib import Path


def test_variable_rank_fit_widens_only_layer1_and_uses_actual_prefix():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_variable_rank_layer1_joint_fit.py"
    ).read_text(encoding="utf-8")
    assert 'protocol.get("rank_schedule_prefix") != [768, 1044]' in text
    assert "layer0.forward_with_cache" in text
    assert "residual_rank=rank" in text
    assert "layer1.mlp_output_projection.weight.copy_(basis)" in text
    assert "for step, row in enumerate(train_rows, start=1)" in text


def test_variable_rank_fit_is_fail_fast_and_checkpoint_is_pass_gated():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_variable_rank_layer1_joint_fit.py"
    ).read_text(encoding="utf-8")
    assert "if passed:" in text
    assert '"native_trajectory_mixed_rank_layers_00_01.safetensors"' in text
    assert '"source_blocks_in_checkpoint": 0' in text
    assert '"phase3_certified": False' in text
