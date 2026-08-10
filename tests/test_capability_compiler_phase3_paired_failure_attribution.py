from abi.capability_compiler_phase3_paired_failure_attribution import evaluator_literals


def test_evaluator_literals_flattens_nested_rules():
    evaluator = {
        "kind": "all_of",
        "rules": [
            {"kind": "contains_all", "values": ["CODE", "Wednesday"]},
            {"kind": "contains_any", "values": ["please", "thank"]},
        ],
    }
    assert evaluator_literals(evaluator) == ["CODE", "Wednesday", "please", "thank"]


def test_regex_has_no_literal_assumption():
    assert evaluator_literals({"kind": "regex", "pattern": "^[A-Z]+$"}) == []
