import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_b20_missing_seed_lineage import _run


def test_run_rejects_unregistered_seed():
    protocol = {"runs": [{"seed": 130363}]}
    assert _run(protocol, 130363)["seed"] == 130363
    with pytest.raises(Phase3Error, match="unregistered"):
        _run(protocol, 104729)
