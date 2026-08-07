import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_teacher_representation_verify import (
    _select_samples,
    _token_span,
    _verify_tensor_structure,
)


def test_tensor_verifier_rejects_wrong_keys_shape_dtype_and_nonfinite() -> None:
    good = {
        "prompt_pooled": torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.float16),
        "response_pooled": torch.tensor([[2.0, 1.0], [6.0, 3.0]], dtype=torch.float16),
    }
    assert _verify_tensor_structure(good, 2, 2)["prompt_pooled"]["shape"] == [2, 2]
    with pytest.raises(Phase3Error):
        _verify_tensor_structure({"prompt_pooled": good["prompt_pooled"]}, 2, 2)
    bad = dict(good)
    bad["response_pooled"] = torch.ones((2, 3), dtype=torch.float16)
    with pytest.raises(Phase3Error):
        _verify_tensor_structure(bad, 2, 2)
    bad = dict(good)
    bad["response_pooled"] = torch.tensor([[1.0, float("nan")], [2.0, 3.0]], dtype=torch.float16)
    with pytest.raises(Phase3Error):
        _verify_tensor_structure(bad, 2, 2)


def test_stratified_sample_is_deterministic_and_offset_span_fails_closed() -> None:
    rows = [
        {"record_id": f"{index:064x}", "capability": capability}
        for capability in ("a", "b")
        for index in range(4)
    ]
    assert _select_samples(rows, 2, 59) == _select_samples(rows, 2, 59)
    assert len(_select_samples(rows, 2, 59)) == 4
    assert _token_span([(0, 1), (1, 3), (3, 4)], 1, 4) == (1, 2)
    with pytest.raises(Phase3Error):
        _token_span([(0, 2), (2, 4)], 1, 4)
