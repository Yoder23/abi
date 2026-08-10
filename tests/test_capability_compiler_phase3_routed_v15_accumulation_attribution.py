from __future__ import annotations

import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_routed_v15_accumulation_attribution import metrics, source_token_id


def test_host_to_source_action_mapping_is_exact() -> None:
    assert source_token_id(2, 32007) == 32007
    assert source_token_id(4, 32007) == 0
    assert source_token_id(32014, 32007) == 32010
    with pytest.raises(Phase3Error): source_token_id(1, 32007)


def test_geometry_metrics_identity() -> None:
    value = metrics(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
    assert value["cosine"] == pytest.approx(1.0)
    assert value["relative_rmse"] == 0.0
