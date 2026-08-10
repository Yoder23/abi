from pathlib import Path


def test_native_attention_oracle_is_read_only_and_uses_exact_teacher_interface():
    text = (Path(__file__).parents[1] / "abi" / "capability_compiler_phase3_native_attention_interface_oracle.py").read_text(encoding="utf-8")
    assert "native_attention, target = dual._teacher_components" in text
    assert "layer.post_attention_norm(native_attention)" in text
    assert '"training_performed": False' in text
    assert '"artifact_written": False' in text
