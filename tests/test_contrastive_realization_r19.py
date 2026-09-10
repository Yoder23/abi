from __future__ import annotations

import hashlib
from dataclasses import astuple

import pytest

from experiments.contrastive_realization_r19.compiler import (
    _one_literal_insertion,
    infer_templates,
)
from experiments.contrastive_realization_r19.hidden_frames import (
    LEXEME_BANK,
    R19HiddenFrameError,
    heldout_rows,
)
from experiments.contrastive_realization_r19.isolated_worker import (
    infer_templates as infer_isolated,
)
from experiments.functional_realization_r18.hidden_frames import LEXEME_BANK as R18_LEXEMES
from experiments.functional_realization_r18.run_public import _control
from experiments.functional_realization_r18.verify import _fill
from experiments.linguistic_realization_r17.frames import (
    EVALUATION_LEXEMES,
    EXTRACTION_LEXEMES,
    SIGNATURES,
)
from experiments.linguistic_realization_r17.frames_v2 import public_rows_v2


def test_structural_polarity_contrast_has_no_hardcoded_marker() -> None:
    assert _one_literal_insertion(
        "X {subject} {verb_base} {object}?", "X {subject} Z {verb_base} {object}?"
    )
    assert not _one_literal_insertion(
        "X {subject} {verb_base} {object}?", "Y {subject} Z {verb_base} {object}?"
    )


def test_contrastive_selection_rejects_noisy_unpaired_majority() -> None:
    records = []
    for row in public_rows_v2("extraction"):
        output = row["expected"]
        if (
            row["signature"] == "question|present|positive|plural"
            and len([item for item in records if item["signature"] == row["signature"]]) < 2
        ):
            output = f"Have {row['slots']['subject']} {row['slots']['verb_past']} {row['slots']['object']}?"
        records.append(
            {
                "record_id": row["record_id"],
                "signature": row["signature"],
                "slots": row["slots"],
                "output": output,
            }
        )
    templates, _ = infer_templates(records)
    assert templates["question|present|positive|plural"] == "Do {subject} {verb_base} {object}?"
    assert all(
        _fill(templates[row["signature"]], row["slots"]) == row["expected"]
        for row in public_rows_v2("evaluation")
    )
    isolated_templates, _ = infer_isolated(records, list(templates))
    assert isolated_templates == templates


def test_mood_permutation_fails_closed_without_package() -> None:
    from experiments.contrastive_realization_r19.compiler import R19CompilerError

    records = [
        {
            "record_id": row["record_id"],
            "signature": row["signature"],
            "slots": row["slots"],
            "output": row["expected"],
        }
        for row in public_rows_v2("extraction")
    ]
    with pytest.raises(R19CompilerError, match="no structure-compatible"):
        infer_templates(_control(records))


def test_hidden_lexical_bank_is_fully_disjoint_from_r17_and_r18() -> None:
    prior = {
        field
        for lexeme in (*EXTRACTION_LEXEMES, *EVALUATION_LEXEMES, *R18_LEXEMES)
        for field in astuple(lexeme)
    }
    current = {field for lexeme in LEXEME_BANK for field in astuple(lexeme)}
    assert len(LEXEME_BANK) == 20
    assert not prior.intersection(current)


def test_hidden_selection_is_committed_deterministic_and_split_disjoint() -> None:
    secret = bytes(range(32))
    secret_hex = secret.hex()
    commitment = hashlib.sha256(secret).hexdigest()
    extraction = heldout_rows(secret_hex, commitment, "extraction")
    evaluation = heldout_rows(secret_hex, commitment, "evaluation")
    assert len(extraction) == 72
    assert len(evaluation) == 48
    assert len({row["record_id"] for row in extraction + evaluation}) == 120
    assert extraction == heldout_rows(secret_hex, commitment, "extraction")
    for signature in SIGNATURES:
        extraction_slots = {
            tuple(sorted(row["slots"].items()))
            for row in extraction
            if row["signature"] == signature
        }
        evaluation_slots = {
            tuple(sorted(row["slots"].items()))
            for row in evaluation
            if row["signature"] == signature
        }
        assert len(extraction_slots) == 3
        assert len(evaluation_slots) == 2
        assert extraction_slots.isdisjoint(evaluation_slots)


def test_hidden_selection_fails_closed_on_bad_secret_or_split() -> None:
    secret = bytes(range(32))
    commitment = hashlib.sha256(secret).hexdigest()
    with pytest.raises(R19HiddenFrameError, match="commitment mismatch"):
        heldout_rows(bytes(reversed(secret)).hex(), commitment, "extraction")
    with pytest.raises(R19HiddenFrameError, match="unknown R19 held-out split"):
        heldout_rows(secret.hex(), commitment, "invalid")
