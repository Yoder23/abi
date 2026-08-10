from pathlib import Path

def test_native_trajectory_layer0_is_explicitly_distillation_classified():
    text=(Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_native_trajectory_full_operator_layer0.py").read_text(encoding="utf-8")
    assert "hidden-state-distillation" in text
    assert "source_blocks_in_checkpoint\":0" in text
