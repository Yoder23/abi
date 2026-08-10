from pathlib import Path


def test_stable_vo_replay_changes_only_numeric_factorization():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_within_head_vo_stable_replay.py"
    ).read_text(encoding="utf-8")
    assert "prior_qr(values.double()" in text
    assert "prior_svd(values.double()" in text
    assert "if weight.dtype == torch.float64" in text
    assert "original.execute(root, protocol_path" in text


def test_stable_vo_replay_preserves_raw_replay_and_writes_no_weights():
    text = (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_within_head_vo_stable_replay.py"
    ).read_text(encoding="utf-8")
    assert 'output / "numeric_replay"' in text
    assert "save_file" not in text
    assert "torch.optim" not in text
