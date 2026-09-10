from __future__ import annotations

from pathlib import Path

from experiments.generative_transfer_r21 import run as base_run
from experiments.layercake_composition_r24.protocol import (
    NAMESPACES,
    SEEDS,
    domain_slug,
    prepared_rows,
    source_splits,
)


def test_r24_source_split_is_fixed_and_prompt_disjoint() -> None:
    root = Path(__file__).resolve().parents[1]
    training, evaluation = source_splits(
        root / "results/factual_semantic_r16/heldout_v2/source_observations.jsonl"
    )
    assert len(training) == len(evaluation) == 48
    assert {row["namespace"] for row in training} == set(NAMESPACES)
    assert {row["namespace"] for row in evaluation} == set(NAMESPACES)
    assert {row["question"] for row in training}.isdisjoint(
        {row["question"] for row in evaluation}
    )
    assert all(
        sum(row["namespace"] == namespace for row in training) == 24
        for namespace in NAMESPACES
    )


def test_r24_prepared_answers_are_lossless_for_frozen_tokenizer() -> None:
    root = Path(__file__).resolve().parents[1]
    training, evaluation = source_splits(
        root / "results/factual_semantic_r16/heldout_v2/source_observations.jsonl"
    )
    api = base_run._layercake(root)
    prepared_training = prepared_rows(training)
    tokenizer = api["LosslessLexemePointerTokenizer"].build_generic(
        prepared_training
    )
    for rows in (prepared_training, prepared_rows(evaluation)):
        for row in rows:
            _, source_lexemes = tokenizer.encode_source(row["prompt"])
            actions = tokenizer.encode_target(
                row["response"],
                copy_lexemes=row["copy_lexemes"],
                source_lexemes=source_lexemes,
            )
            assert tokenizer.decode_actions(actions, source_lexemes).decode(
                "utf-8"
            ) == row["response"]


def test_r24_identifiers_are_fixed() -> None:
    assert SEEDS == (24021, 24022, 24023)
    assert domain_slug("chemistry/periodic-table") == "chemistry-periodic-table"
    assert domain_slug("geography/national-capitals") == "geography-national-capitals"
