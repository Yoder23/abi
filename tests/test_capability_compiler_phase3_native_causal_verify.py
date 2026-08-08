import torch

from abi.capability_compiler_phase3_native_causal_verify import _select, _verify_offsets


def test_offsets_and_selection_are_deterministic() -> None:
    offsets = torch.tensor([0, 2, 5], dtype=torch.int64)
    _verify_offsets(offsets, [2, 3], 5)
    rows = [
        {"record_id": f"{capability}-{index}", "capability": capability}
        for capability in [f"cap-{value}" for value in range(14)]
        for index in range(3)
    ]
    first = _select(rows, 92, 2)
    assert first == _select(rows, 92, 2)
    assert len(first) == 28
