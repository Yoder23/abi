import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_action_aligned_verify import _select, _verify_offsets


def test_offsets_and_stratified_selection_fail_closed() -> None:
    _verify_offsets(torch.tensor([0, 2, 5], dtype=torch.int64), [2, 3], 5)
    with pytest.raises(Phase3Error):
        _verify_offsets(torch.tensor([0, 3, 5], dtype=torch.int64), [2, 3], 5)
    rows = [{"record_id": f"{index:064x}", "capability": capability} for capability in [f"c{x}" for x in range(14)] for index in range(3)]
    assert len(_select(rows, 68, 2)) == 28
