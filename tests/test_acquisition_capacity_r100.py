import copy

import pytest

from experiments.acquisition_capacity_r100 import core as c
from experiments.acquisition_capacity_r100 import run


def test_prospective_population():
    rows = c.development_catalog()
    assert rows == c.development_catalog()
    assert len(rows) == 48 and len({r["id"] for r in rows}) == 48
    assert all(sum(r["task"] == t for r in rows) == 6 for t in c.TASKS)
    old = (
        c.r99.development_catalog()
        + c.previous.original.catalog("train")
        + c.previous.original.catalog("development")
    )
    assert not {r["prompt"] for r in rows} & {r["prompt"] for r in old}
    assert not {r["code"] for r in rows} & {r["code"] for r in old}
    assert all(fact in row["prompt"] for row in rows for fact in row["plan"]["facts"])
    assert all(value in row["prompt"] for row in rows for value in row["required"])


def test_only_scope_changes_from_response_control():
    row = c.development_catalog()[0]
    assert c.SEED == c.r99.SEED and c.STEPS == 384
    assert c.model_prompt(row, "full_trunk") == c.model_prompt(row, "distillation")
    assert c.target(row, "A response.", "full_trunk") == "A response."
    assert c.checkpoint("lora") == c.baseline.OUT / "lora"
    assert c.checkpoint("parent") == c.PARENT
    with pytest.raises(ValueError):
        c.checkpoint("unregistered")


def test_paired_interval_and_missing_control():
    good = [{"id": str(i), "screen": {"mechanical_pass": True}} for i in range(24)]
    bad = [{"id": str(i), "screen": {"mechanical_pass": False}} for i in range(24)]
    assert run.paired(good, bad) == {"difference": 1, "ci95": [1, 1]}
    assert run.paired(good, good) == {"difference": 0, "ci95": [0, 0]}
    with pytest.raises(RuntimeError):
        run.paired(good, bad[:-1])
    with pytest.raises(RuntimeError):
        run.paired([], [])


class Tokenizer:
    def __len__(self):
        return 2

    def decode(self, tokens, **kwargs):
        return "No approved budget is supplied for NOTETEST."


def observation():
    reference = {
        "id": "test",
        "task": "abstention",
        "code": "NOTETEST",
        "required": ["NOTETEST"],
        "prompt": "Project NOTETEST has no budget.",
    }
    text = Tokenizer().decode([0])
    row = {
        "id": "test",
        "task": "abstention",
        "token_ids": [0],
        "raw_output": text,
        "output": text,
        "plan_valid": True,
        "screen": c.screen(reference, text),
    }
    return reference, row


def test_recomputable_observation():
    reference, row = observation()
    run.validate_generation([row], [reference], "full_trunk", Tokenizer())


@pytest.mark.parametrize(
    "field,value",
    [
        ("output", "made up"),
        ("raw_output", "made up"),
        ("token_ids", [99]),
        ("token_ids", [True]),
        ("task", "summarization"),
        ("plan_valid", False),
        ("screen", {}),
    ],
)
def test_hostile_observation_rejected(field, value):
    reference, row = observation()
    row = copy.deepcopy(row)
    row[field] = value
    with pytest.raises(RuntimeError):
        run.validate_generation([row], [reference], "full_trunk", Tokenizer())


def test_missing_and_duplicate_rows_rejected():
    reference, row = observation()
    for rows in ([], [row, row]):
        with pytest.raises(RuntimeError):
            run.validate_generation(rows, [reference], "full_trunk", Tokenizer())
