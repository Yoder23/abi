import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase3_route_isolated_replication import hierarchical_bootstrap


def _seed(value):
    return {"a": [value] * 100, "b": [value] * 100}


def test_hierarchical_bootstrap_is_deterministic_and_positive():
    first = hierarchical_bootstrap([_seed(1), _seed(1), _seed(1)], replicates=1000, seed=7)
    second = hierarchical_bootstrap([_seed(1), _seed(1), _seed(1)], replicates=1000, seed=7)
    assert first == second
    assert first["lower_95"] == 1.0
    assert first["prompt_observations"] == 600


def test_hierarchical_bootstrap_rejects_wrong_prompt_depth():
    invalid = _seed(1)
    invalid["a"] = invalid["a"][:-1]
    with pytest.raises(Phase3Error, match="prompt depth"):
        hierarchical_bootstrap([_seed(1), invalid, _seed(1)], replicates=1000, seed=7)
