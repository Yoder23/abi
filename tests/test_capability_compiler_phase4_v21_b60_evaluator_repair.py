import pytest
from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_v21_b60_evaluator_repair import expected_changed_rows


def test_seed_specific_change_contract():
    assert expected_changed_rows(104729)==1
    assert expected_changed_rows(130363)==0
    assert expected_changed_rows(155921)==0


def test_unregistered_seed_fails_closed():
    with pytest.raises(Phase3Error): expected_changed_rows(1)
