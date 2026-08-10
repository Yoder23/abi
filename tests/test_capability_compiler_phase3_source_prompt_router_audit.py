import pytest
import torch

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_source_prompt_router_audit import (
    FORMAT,
    _route,
    _solve_dual_ridge,
)


def test_source_prompt_router_format_is_versioned() -> None:
    assert "source-prompt-router" in FORMAT
    assert FORMAT.endswith("/1")


def test_source_prompt_route_has_only_two_specialists() -> None:
    specialists = ("abstention", "conversation")
    assert _route("abstention", specialists) == "abstention"
    assert _route("grammar", specialists) == "generic"


def test_dual_ridge_rejects_observation_drift() -> None:
    with pytest.raises(Phase3Error, match="observations changed"):
        _solve_dual_ridge(torch.zeros(2, 3), torch.zeros(1, 2), 1e-4)
