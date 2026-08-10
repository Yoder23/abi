from pathlib import Path


def test_rank1044_audit_uses_actual_prefix_and_existing_sparse_features():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_rank1044_coefficient_audit.py"
    ).read_text(encoding="utf-8")
    assert "layer0.forward_with_cache" in text
    assert "layer1.post_attention_norm(attention)" in text
    assert "layer1.sparse_gate_up_projection.weight.detach()" in text
    assert "closed.solve_ridge" in text
    assert "int(protocol.get(\"rank\", 0)) != 1044" in text


def test_rank1044_audit_cannot_write_model_or_train_gradients():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_rank1044_coefficient_audit.py"
    ).read_text(encoding="utf-8")
    assert "torch.optim" not in text
    assert "save_file" not in text
    assert '"gradient_training_performed": False' in text
    assert '"artifact_written": False' in text
