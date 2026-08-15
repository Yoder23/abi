import pytest

from abi.capability_compiler_phase3 import Phase3Error
from abi.capability_compiler_phase4_b50_l1_conformance_verify import (
    FORMAT,
    SEEDS,
    independently_conform,
    verify_output_row,
)


def _probe(capability="abstention"):
    return {
        "canonical_capability": capability,
        "evaluator": {"kind": "contains_any", "values": ["unknown"]},
    }


def test_verifier_contract_is_separate():
    assert FORMAT == "abi-capability-compiler-phase4-b50-l1-conformance-verify/1"
    assert SEEDS == (104729, 130363, 155921)


def test_independent_rule_is_route_bounded():
    original = "The answer cannot be known from the information given."
    assert independently_conform(original, "abstention") == (
        "The answer is unknown from the information given.",
        1,
    )
    assert independently_conform(original, "grammar") == (original, 0)


def test_row_verifier_rejects_output_mutation():
    source = {"probe_id": "p1", "output": "It cannot be known."}
    observed = {
        "probe_id": "p1",
        "capability": "abstention",
        "output": "It is unknown.",
        "functional_pass_v1": True,
        "functional_pass_v2": True,
        "repetition_collapse_v2": False,
        "conformance_changed": True,
    }
    expected, changed = verify_output_row(source, observed, _probe())
    assert expected == observed
    assert changed["semantic_v2_preserved"]
    mutated = dict(observed, output="A different answer.")
    with pytest.raises(Phase3Error, match="conformed output changed"):
        verify_output_row(source, mutated, _probe())


def test_row_verifier_rejects_multiple_replacements():
    source = {
        "probe_id": "p2",
        "output": "It cannot be known; it cannot be known.",
    }
    observed = {
        "probe_id": "p2",
        "capability": "abstention",
        "output": "It is unknown; it is unknown.",
        "functional_pass_v1": True,
        "functional_pass_v2": True,
        "repetition_collapse_v2": False,
        "conformance_changed": True,
    }
    with pytest.raises(Phase3Error, match="multiple replacements"):
        verify_output_row(source, observed, _probe())
