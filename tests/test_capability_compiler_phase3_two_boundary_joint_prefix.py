from pathlib import Path


def test_two_boundary_joint_prefix_locks_both_native_boundaries():
    text=(Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_two_boundary_joint_prefix.py").read_text(encoding="utf-8")
    assert "loss=losses[0]+losses[1]" in text
    assert 'gates[f"layer{index}_mean_cosine"]' in text
    assert "if passed:" in text
    assert '"source_blocks_in_checkpoint":0' in text
