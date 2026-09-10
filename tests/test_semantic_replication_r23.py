from __future__ import annotations

import json
from pathlib import Path

from experiments.generative_transfer_r21.protocol import WordLabeler
from experiments.semantic_replication_r23.protocol import (
    HIDDEN_INSTRUCTIONS,
    hidden_rows,
    seed_commitment,
    semantic_score,
    validate_public_lexicon,
)


def test_semantic_summary_accepts_supported_paraphrase_but_not_wrong_number() -> None:
    row = {
        "task": "summary",
        "prompt": "case-41 latency 3 day 8",
        "slots": {
            "field_a": "sensor case-41 remained stable",
            "field_b": "latency fell by 3 milliseconds",
            "field_c": "review is scheduled for day 8",
        },
    }
    good = (
        "Summary: Sensor case-41 showed stability, latency decreased by 3 "
        "milliseconds, and a review is planned for day 8."
    )
    assert semantic_score(row, good, good)["functional_pass"]
    bad = good.replace("day 8", "day 9")
    assert not semantic_score(row, bad, good)["functional_pass"]
    unstable = good.replace("showed stability", "was unstable")
    assert not semantic_score(row, unstable, good)["functional_pass"]
    swapped = good.replace("by 3 milliseconds", "by 8 milliseconds").replace(
        "day 8", "day 3"
    )
    assert not semantic_score(row, swapped, good)["functional_pass"]


def test_hidden_rows_are_deterministic_unique_and_labelable() -> None:
    seed = "12" * 32
    rows = hidden_rows(seed)
    assert seed_commitment(seed) == seed_commitment(seed)
    assert len(rows) == len({row["record_id"] for row in rows}) == 120
    assert set(HIDDEN_INSTRUCTIONS) == {row["task"] for row in rows}
    root = Path(__file__).resolve().parents[1]
    labeler = WordLabeler(
        json.loads(
            (
                root
                / "results/generative_transfer_r21/public_v5_bakeoff/engine/labeler.json"
            ).read_text(encoding="utf-8")
        )
    )
    assert all(labeler.predict(row["instruction"]) == row["task"] for row in rows)


def test_every_semantic_marker_is_supported_by_public_corpus() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / (
        "results/generative_transfer_r21/public_v3_source/source_observations.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    counts = validate_public_lexicon(rows)
    assert counts and min(counts.values()) >= 1
