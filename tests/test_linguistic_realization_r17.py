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
from experiments.linguistic_realization_r17.isolated_worker import _template
from experiments.linguistic_realization_r17.package import PACKAGE_FORMAT, realize


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
