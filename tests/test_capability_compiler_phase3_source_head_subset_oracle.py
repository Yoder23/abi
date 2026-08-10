from pathlib import Path


def test_head_subset_is_train_derived_once_without_head_count_sweep():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_source_head_subset_oracle.py"
    ).read_text(encoding="utf-8")
    assert "source_attention.o_proj.register_forward_pre_hook" in text
    assert 'torch.zeros(heads, heads, dtype=torch.float64' in text
    assert "selected_heads, selection_curve = _greedy_subset" in text
    assert '"head_count_sweep_performed": False' in text
    assert "layer0.forward_with_cache" in text


def test_head_subset_does_not_copy_or_promote_source_weights():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_source_head_subset_oracle.py"
    ).read_text(encoding="utf-8")
    assert "save_file" not in text
    assert "torch.optim" not in text
    assert '"source_blocks_promoted": 0' in text
    assert '"artifact_written": False' in text
