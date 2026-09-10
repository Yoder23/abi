from __future__ import annotations

import json
from pathlib import Path

from experiments.linguistic_realization_r17.frames import (
    EVALUATION_LEXEMES,
    EXTRACTION_LEXEMES,
    SIGNATURES,
    expected,
    public_rows,
)
from experiments.linguistic_realization_r17.frames_v2 import public_rows_v2
from experiments.linguistic_realization_r17.isolated_worker import _template
from experiments.linguistic_realization_r17.package import PACKAGE_FORMAT, realize
from experiments.linguistic_realization_r17.verify_source import verify_source


def test_public_r17_rows_are_balanced_and_lexically_disjoint() -> None:
    extraction = public_rows("extraction")
    evaluation = public_rows("evaluation")
    assert len(SIGNATURES) == 24
    assert len(extraction) == 72
    assert len(evaluation) == 48
    assert {row["signature"] for row in extraction} == set(SIGNATURES)
    assert {row["signature"] for row in evaluation} == set(SIGNATURES)
    assert not (
        {item.singular_subject for item in EXTRACTION_LEXEMES}
        & {item.singular_subject for item in EVALUATION_LEXEMES}
    )
    assert all(row["expected"] == expected(row) for row in extraction + evaluation)


def test_template_extraction_and_generic_realization() -> None:
    rows = public_rows("extraction")
    templates = {}
    for row in rows:
        learned = _template(row["expected"], row["slots"])
        prior = templates.setdefault(row["signature"], learned)
        assert prior == learned
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/realization",
        "templates": templates,
    }
    for row in public_rows("evaluation"):
        assert realize(package, row["signature"], row["slots"]) == row["expected"]
        assert realize(None, row["signature"], row["slots"]) is None


def test_public_protocol_does_not_claim_full_english() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "experiments/linguistic_realization_r17/PUBLIC_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    assert "would not establish unrestricted English" in text
    assert "superiority to LoRA or distillation" in " ".join(text.split())
    json.dumps(public_rows("evaluation"), sort_keys=True)


def test_v2_changes_only_prompt_and_negative_question_surface_order() -> None:
    v1 = public_rows("evaluation")
    v2 = public_rows_v2("evaluation")
    assert len(v1) == len(v2) == 48
    for original, repaired in zip(v1, v2, strict=True):
        ignored = {"prompt", "expected"}
        assert {key: value for key, value in original.items() if key not in ignored} == {
            key: value for key, value in repaired.items() if key not in ignored
        }
        mood, _tense, polarity, _number = original["signature"].split("|")
        if mood == "question" and polarity == "negative":
            assert original["expected"] != repaired["expected"]
            assert " not " in repaired["expected"]
        else:
            assert original["expected"] == repaired["expected"]
        assert repaired["expected"] not in repaired["prompt"]


def test_completed_public_source_failures_recompute_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    v1 = verify_source(root / "results/linguistic_realization_r17/public_v3", "v1")
    v2 = verify_source(root / "results/linguistic_realization_r17/public_v4_interface_v2", "v2")
    assert v1["metrics"]["extraction_source_exact"] == 38
    assert v1["metrics"]["evaluation_source_exact"] == 28
    assert v2["metrics"]["extraction_source_exact"] == 67
    assert v2["metrics"]["evaluation_source_exact"] == 47
    assert v1["compiler_invoked"] is v2["compiler_invoked"] is False
