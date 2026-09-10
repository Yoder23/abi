from __future__ import annotations

from experiments.contrastive_realization_r19.compiler import (
    _one_literal_insertion,
    infer_templates,
)
from experiments.contrastive_realization_r19.isolated_worker import (
    infer_templates as infer_isolated,
)
from experiments.functional_realization_r18.run_public import _control
from experiments.functional_realization_r18.verify import _fill
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
    import pytest

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
