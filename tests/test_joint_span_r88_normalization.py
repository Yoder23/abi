from __future__ import annotations

import random
from pathlib import Path

from transformers import AutoTokenizer

from experiments.joint_span_r88.train_candidate_v1 import _rename_row


ROOT = Path(__file__).resolve().parents[1]


def test_r88_renaming_is_deterministic_bijective_and_validation_disjoint() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "results/role_invariant_choice_r76/candidate_v1",
        local_files_only=True,
    )
    row = {
        "record_id": "unit-record",
        "prompt": (
            "Rule 1: Every KAV103343 is a LEM103343. Rule 2: Every LEM103343 "
            "is a NUR103343. Start: OBJ103343 is a KAV103343. Apply exactly "
            "the two supplied transitions to the start item. Candidate codes in "
            "random order: KAV103343 | NUR103343 | LEM103343. Return only the "
            "final destination code, with no explanation."
        ),
        "response": "NUR103343",
        "teacher_tokens": 8,
    }
    first = _rename_row(tokenizer, row, random.Random(88_001))
    second = _rename_row(tokenizer, row, random.Random(88_001))

    assert first == second
    assert first["base_record_id"] == "unit-record"
    assert len(first["valid_starts"]) == 2
    assert 1 <= first["span_length"] <= 16
    numeric = int(first["record_id"].split(":")[1])
    assert 200_000 <= numeric < 900_000
    assert not 995_000 <= numeric <= 996_399
