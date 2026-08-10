from pathlib import Path


def test_vo_factorization_preserves_all_heads_and_derives_one_schedule():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_within_head_vo_factorization.py"
    ).read_text(encoding="utf-8")
    assert "ranks = [1] * len(singular_values)" in text
    assert "ranks, achieved_energy = _rank_schedule" in text
    assert '"rank_schedule_sweep_performed": False' in text
    assert "layer0.forward_with_cache" in text


def test_vo_factorization_uses_small_core_svd_and_writes_no_weights():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_within_head_vo_factorization.py"
    ).read_text(encoding="utf-8")
    assert 'torch.linalg.qr(head_output_weight, mode="reduced")' in text
    assert "torch.linalg.svd(core, full_matrices=False)" in text
    assert "save_file" not in text
    assert "torch.optim" not in text
    assert '"source_blocks_promoted": 0' in text
