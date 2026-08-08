import torch

from abi.capability_compiler_phase3_native_causal_core import _alignment_batch, _cosine_loss


def test_native_alignment_joins_non_eos_targets() -> None:
    batch = [{"record_id": "r", "target_actions": [4, 5, 2]}]
    tensors = {"target_values": torch.ones((2, 192), dtype=torch.float16), "target_offsets": torch.tensor([0, 2], dtype=torch.int64)}
    teacher, mask = _alignment_batch(batch, {"r": 0}, tensors, 3, torch.device("cpu"))
    assert teacher.shape == (1, 3, 192)
    assert mask.tolist() == [[True, True, False]]
    assert _cosine_loss(teacher, teacher, mask).item() < 1e-6
