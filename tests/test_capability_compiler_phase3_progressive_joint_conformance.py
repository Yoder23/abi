from pathlib import Path


def test_progressive_joint_uses_replacement_prefix_and_fail_fast_checkpoint():
    text=(Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_progressive_joint_conformance.py").read_text(encoding="utf-8")
    assert "for prefix in prefix_layers" in text
    assert "for source_index in range(layer_index+1)" in text
    assert "if passed:" in text
    assert '"source_blocks_in_checkpoint":0' in text
