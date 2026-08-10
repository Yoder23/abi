from pathlib import Path


def _text() -> str:
    return (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_factorized_layer1_dtype_replay.py"
    ).read_text(encoding="utf-8")


def test_dtype_replay_changes_only_sparse_feature_conformance():
    text = _text()
    assert "result.dtype == torch.bfloat16" in text
    assert "result.shape[-1] == 384" in text
    assert "return result.float()" in text
    assert 'output / "dtype_replay"' in text


def test_dtype_replay_reuses_original_experiment_and_does_not_train():
    text = _text()
    assert "original.execute(root, protocol_path" in text
    assert "torch.optim" not in text
    assert "save_file" not in text
