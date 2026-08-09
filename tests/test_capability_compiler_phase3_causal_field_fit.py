import torch
from abi.capability_compiler_phase3_causal_field_fit import _packed


def test_packed_causal_positions_predict_targets_after_bos():
    rows = [{"source_ids": [4, 5], "target_actions": [6, 2]}]
    inputs, positions, targets = _packed(rows, torch.device("cpu"))
    assert inputs.tolist() == [[4, 5, 1, 6]]
    assert positions.tolist() == [[2, 3]]
    assert targets.tolist() == [[6, 2]]
