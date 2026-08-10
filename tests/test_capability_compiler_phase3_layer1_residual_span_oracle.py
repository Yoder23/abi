from pathlib import Path


def test_layer1_span_oracle_uses_replacement_prefix_and_direct_coefficients():
    text=(Path(__file__).parents[1]/"abi"/"capability_compiler_phase3_layer1_residual_span_oracle.py").read_text(encoding="utf-8")
    assert "layer0.forward_with_cache" in text
    assert "coefficients=(residual-mean)@basis" in text
    assert '"training_performed":False' in text
    assert '"artifact_written":False' in text
