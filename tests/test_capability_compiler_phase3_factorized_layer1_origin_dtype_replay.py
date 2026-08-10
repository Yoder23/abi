from pathlib import Path


def _text() -> str:
    return (
        Path(__file__).parents[1]
        / "abi"
        / "capability_compiler_phase3_factorized_layer1_origin_dtype_replay.py"
    ).read_text(encoding="utf-8")


def test_origin_replay_conforms_the_single_sparse_creation_boundary():
    text = _text()
    assert "prior_silu = original.F.silu" in text
    assert "result.shape[-1] == 384" in text
    assert "return result.float()" in text
    assert 'output / "origin_dtype_replay"' in text


def test_origin_replay_reuses_original_without_training_or_writes():
    text = _text()
    assert "original.execute(root, protocol_path" in text
    assert "torch.optim" not in text
    assert "save_file" not in text
