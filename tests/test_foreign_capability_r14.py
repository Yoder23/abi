from __future__ import annotations

import random

import pytest

from experiments.foreign_capability_r14.capability import (
    behavior_space_size,
    generate_rows,
    order_counterfactual_rows,
    public_capability,
)
from experiments.foreign_capability_r14.extractor import (
    R14ExtractorError,
    candidate_operations,
    extract_transition,
)
from experiments.native_isa_r11.core import transition_accuracy


def test_non_exhaustive_rows_are_unique_and_large_space() -> None:
    capability = public_capability(14001)
    training = generate_rows(capability, split="train", rows=500, depths=range(2, 9), seed=14002)
    excluded = {row["program_key"] for row in training}
    heldout = generate_rows(
        capability,
        split="heldout",
        rows=1000,
        depths=range(10, 19),
        seed=14003,
        excluded_keys=excluded,
    )
    assert behavior_space_size(range(10, 19)) > 1_000_000_000
    assert not {row["program_key"] for row in training} & {row["program_key"] for row in heldout}
    assert all(row["depth"] >= 10 for row in heldout)


def test_counterfactual_pairs_change_only_order_and_answer() -> None:
    capability = public_capability(14004)
    rows = order_counterfactual_rows(capability, pairs=20, depth=12, seed=14005)
    assert len(rows) == 40
    for left, right in zip(rows[::2], rows[1::2]):
        assert left["start"] == right["start"]
        assert sorted(left["program"]) == sorted(right["program"])
        assert left["program"] != right["program"]
        assert left["answer"] != right["answer"]


def test_extractor_recovers_program_without_answers() -> None:
    capability = public_capability(14006)
    rows = generate_rows(capability, split="query", rows=96, depths=range(3, 9), seed=14007)
    generator = random.Random(14008)
    observations = []
    for row in rows:
        probability = [0.01 / 7] * 8
        probability[row["answer"]] = 0.99
        # Inject bounded source-like errors; probabilities remain informative.
        if generator.random() < 0.1:
            wrong = (row["answer"] + 1) % 8
            probability[row["answer"]], probability[wrong] = 0.20, 0.7914285714285714
        observations.append(
            {
                "row_id": row["row_id"],
                "prompt_sha256": row["prompt_sha256"],
                "start": row["start"],
                "program": row["program"],
                "canonical_probabilities": probability,
            }
        )
    transition, receipt = extract_transition(observations)
    assert receipt["answers_consumed"] == 0
    assert receipt["atomic_observations"] == 0
    assert transition_accuracy(transition, rows) == 1.0


def test_extractor_fails_closed_on_answer_field() -> None:
    with pytest.raises(R14ExtractorError):
        extract_transition(
            [
                {
                    "row_id": "x",
                    "prompt_sha256": "y",
                    "start": 0,
                    "program": [0, 1],
                    "canonical_probabilities": [0.125] * 8,
                    "answer": 0,
                }
            ]
        )


def test_candidate_family_is_fixed_and_nontrivial() -> None:
    candidates = candidate_operations()
    assert len(candidates) > 20_000
    assert len(candidates) == len(set(candidates))
