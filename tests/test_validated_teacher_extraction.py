from __future__ import annotations

from abi.validated_teacher_extraction import (
    attempt_passes,
    build_repair_prompt,
    evaluator_feedback,
)


def _evaluator():
    return {
        "kind": "all_of",
        "rules": [
            {"kind": "contains_all", "values": ["alpha", "beta"]},
            {"kind": "contains_none", "values": ["wrong"]},
            {"kind": "maximum_characters", "value": 40},
        ],
        "prompt_contract_sha256": "0" * 64,
    }


def test_feedback_flattens_only_closed_evaluator_requirements():
    feedback = evaluator_feedback(_evaluator())
    assert len(feedback) == 3
    assert "alpha" in feedback[0]
    assert "wrong" in feedback[1]
    assert "40" in feedback[2]


def test_repair_prompt_binds_task_prior_answer_and_requirements():
    prompt = build_repair_prompt(
        original_prompt="Use alpha and beta.",
        prior_output="wrong",
        evaluator=_evaluator(),
    )
    assert "<original_task>" in prompt
    assert "Use alpha and beta." in prompt
    assert "<prior_answer>" in prompt
    assert "wrong" in prompt
    assert "Return only the corrected final answer" in prompt


def test_attempt_requires_eos_as_well_as_functional_pass():
    length_sample = {
        "output": "alpha beta",
        "finish_reason": "length",
    }
    eos_sample = {
        "output": "alpha beta",
        "finish_reason": "eos_token",
    }
    assert attempt_passes(length_sample, _evaluator())[0] is False
    assert attempt_passes(eos_sample, _evaluator())[0] is True


def test_exact_feedback_exposes_only_preregistered_expected_target():
    feedback = evaluator_feedback({"kind": "exact", "value": "First: a"})
    assert feedback == [
        'The entire answer must equal this exact text: "First: a"'
    ]
