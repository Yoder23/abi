import torch

from abi import capability_compiler_phase3_routed_v16_trajectory_retargeting_replay as replay


def test_runtime_repair_exposes_weight_dtype_only_while_active(monkeypatch, tmp_path):
    observed = {}

    def fake_execute(root, protocol, output):
        layer = torch.nn.Linear(2, 3)
        observed["dtype"] = layer.dtype
        return {"status": "SENTINEL"}

    monkeypatch.setattr(replay.original, "execute", fake_execute)
    assert not hasattr(torch.nn.Linear, "dtype")
    assert replay.execute(tmp_path, tmp_path / "p.json", tmp_path / "o") == {"status": "SENTINEL"}
    assert observed["dtype"] == torch.float32
    assert not hasattr(torch.nn.Linear, "dtype")
