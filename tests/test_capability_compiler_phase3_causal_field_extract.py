import torch

from abi.capability_compiler_phase3_causal_field_extract import _topk_field


def test_topk_field_uses_full_normalizer_and_allowed_ids_only():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 8.0]])
    ids, probabilities, residual = _topk_field(logits, top_k=2, allowed_vocabulary=4)
    assert ids.dtype == torch.uint16
    assert ids.tolist() == [[0, 1]]
    assert probabilities.dtype == torch.float16
    assert float(residual[0]) > 0.95
    assert abs(float(probabilities.sum() + residual[0]) - 1.0) < 1e-3


def test_locked_probability_field_tensor_payload():
    positions = 432_371
    payload = positions * (32 * 2 + 32 * 2 + 2) + 14_001 * 8
    assert payload == 56_320_238
