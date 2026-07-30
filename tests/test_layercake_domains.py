from __future__ import annotations

import pytest

from abi.layercake_domains import (
    DomainConformanceError,
    _Sampler,
    _candidate_copy_lexemes,
    _deterministic_math_rows,
)


def test_python_copy_lexeme_uses_declared_function_identity():
    prompt = "Define calculate_071(a, b) and return a - b."
    output = "def calculate_071(a, b):\n    return a - b"
    assert _candidate_copy_lexemes(
        prompt,
        output,
        {
            "kind": "python_function_expression",
            "function_name": "calculate_071",
        },
    ) == ["calculate_071"]


def test_generic_copy_lexeme_is_exact_and_deterministic():
    prompt = "Name the chemical element with atomic number 15."
    output = "The chemical element is Phosphorus."
    assert _candidate_copy_lexemes(prompt, output, {}) == ["chemical"]


def test_prompt_only_pointer_sentinel_does_not_change_source_answer():
    assert _candidate_copy_lexemes(
        "Solve 20 plus 48.",
        "68",
        {"kind": "numeric_equal", "value": 68},
    ) == ["Solve"]


def test_copy_lexeme_fails_closed_without_any_unique_source_lexeme():
    with pytest.raises(DomainConformanceError, match="no unique"):
        _candidate_copy_lexemes("x x", "y", {})


def test_deterministic_math_closure_is_exact_and_teacher_free():
    repair = {
        "kind": "deterministic_elementary_algebra_closure",
        "addend_inclusive_range": [1, 2],
        "solution_inclusive_range": [3, 4],
        "generated_rows": 4,
    }
    rows = _deterministic_math_rows(repair)
    assert len(rows) == 4
    assert rows[0]["prompt"] == (
        "Solve x + 1 = 4. Give only the numerical value of x."
    )
    assert rows[0]["response"] == "The numerical value of x is 3."
    assert all(row["teacher_tokens"] == 0 for row in rows)
    assert all(row["derived_without_teacher"] for row in rows)


def test_sampler_crosses_epoch_boundaries_without_dropping_tail():
    rows = [{"id": value} for value in range(3)]
    sampler = _Sampler(rows, batch_size=5, seed=71)
    first = sampler.next()
    assert len(first) == 5
    assert {row["id"] for row in first[:3]} == {0, 1, 2}


def test_sampler_rejects_empty_rows_and_nonpositive_batch():
    with pytest.raises(DomainConformanceError, match="invalid training batch"):
        _Sampler([], batch_size=1, seed=71)
    with pytest.raises(DomainConformanceError, match="invalid training batch"):
        _Sampler([{"id": 1}], batch_size=0, seed=71)
