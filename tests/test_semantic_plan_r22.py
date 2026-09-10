from __future__ import annotations

from experiments.generative_transfer_r21.protocol import training_rows
from experiments.semantic_plan_r22.protocol import (
    NORMALIZATION_SYSTEM,
    fields_verbatim_once,
    normalization_prompt,
    normalization_request_sha256,
)


def test_normalization_request_binds_plan_raw_response_and_instruction() -> None:
    row = {**training_rows()[0], "teacher_output": "Preserved raw teacher response."}
    prompt = normalization_prompt(row)
    assert row["instruction"] in prompt
    assert row["teacher_output"] in prompt
    assert all(value in prompt for value in row["slots"].values())
    assert "exactly once" in NORMALIZATION_SYSTEM
    assert len(normalization_request_sha256(row)) == 64


def test_fields_verbatim_once_is_ordered_and_fail_closed() -> None:
    row = training_rows()[0]
    values = list(row["slots"].values())
    assert fields_verbatim_once(row, ". ".join(values))
    assert not fields_verbatim_once(row, ". ".join(reversed(values)))
    assert not fields_verbatim_once(row, ". ".join(values + values[:1]))
    assert not fields_verbatim_once(row, ". ".join(values[:2]))
