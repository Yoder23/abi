import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_clarification_route_replicate import _run


def test_run_accepts_only_registered_seed():
    protocol = {"runs": [{"budget": "B40", "seed": 104729}]}
    assert _run(protocol, 104729)["budget"] == "B40"
    with pytest.raises(Phase3Error, match="unregistered"):
        _run(protocol, 155921)
