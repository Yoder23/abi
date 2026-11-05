import copy
from types import SimpleNamespace

import pytest

from experiments.execution_continuation_r101 import core as c


class Tokenizer:
    eos_token_id = 256

    def __len__(self):
        return 257

    def encode(self, text, **kwargs):
        return [ord(ch) for ch in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i != 256)

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=self.encode(text), offset_mapping=[(i, i+1) for i in range(len(text))])


def example():
    row = {"id": "r101-fixture", "task": "abstention", "split": "train",
           "required": ["NOTEZ"], "prompt": "No budget for NOTEZ.", "code": "NOTEZ"}
    target = "The budget for NOTEZ is unknown."
    tok = Tokenizer()
    encoded = tok(target)
    cut = len(encoded.input_ids) // 2
    record = {"id": row["id"], "task": row["task"], "split": row["split"],
              "cut_percent": 50, "supplied_ids": encoded.input_ids[:cut],
              "generated_ids": encoded.input_ids[cut:], "output": target,
              "tail": target[cut:], "stop_reason": "eos",
              "eligible_literals": c.eligible_literals(row, target, encoded.offset_mapping, cut),
              "screen": c.old.screen(row, target)}
    masks = c.fit_audit.token_masks(row, target, encoded.offset_mapping, "full_trunk")
    forced = {"id": row["id"], "task": row["task"], "split": row["split"],
              "target_ids": encoded.input_ids + [256], "argmax_ids": encoded.input_ids + [256],
              "nll": [0.1] * (len(target)+1), "response_mask": masks[0], "fact_mask": masks[1]}
    return row, target, tok, record, forced


def test_recomputes_valid_evidence():
    row, target, tok, record, forced = example()
    c.validate_generation(record, row, target, tok, 50)
    c.validate_forced(forced, row, target, tok)


@pytest.mark.parametrize("key,value", [
    ("output", "fabricated"), ("tail", "fabricated"), ("screen", {"mechanical_pass": True}),
    ("eligible_literals", ["invented"]), ("supplied_ids", []), ("generated_ids", [True]),
    ("generated_ids", [999]), ("generated_ids", [256]), ("cut_percent", 25),
    ("stop_reason", "budget"), ("task", "rewriting"),
])
def test_hostile_generation_rejected(key, value):
    row, target, tok, record, _ = example()
    record[key] = value
    with pytest.raises(RuntimeError):
        c.validate_generation(record, row, target, tok, 50)


@pytest.mark.parametrize("key,value", [
    ("target_ids", []), ("argmax_ids", []), ("response_mask", []),
    ("split", "development"), ("task", "rewriting"),
])
def test_hostile_forced_rejected(key, value):
    row, target, tok, _, forced = example()
    forced[key] = value
    with pytest.raises(RuntimeError):
        c.validate_forced(forced, row, target, tok)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_bad_nll_rejected(bad):
    row, target, tok, _, forced = example()
    forced["nll"][0] = bad
    with pytest.raises(RuntimeError):
        c.validate_forced(forced, row, target, tok)


def test_no_credit_for_pasted_or_partial_facts():
    row = {"required": ["Alpha", "Beta", "Gamma", "absent"]}
    target = "Alpha Beta Gamma"
    offsets = [(i, i+1) for i in range(len(target))]
    assert c.eligible_literals(row, target, offsets, 0) == ["Alpha", "Beta", "Gamma"]
    assert c.eligible_literals(row, target, offsets, 2) == ["Beta", "Gamma"]
    assert c.eligible_literals(row, target, offsets, 7) == ["Gamma"]
    assert c.eligible_literals(row, target, offsets, len(target)) == []
    assert not c.literal_present("Alpha", "Alphabet")


def test_coverage_fails_closed():
    population = [{"id": "one"}, {"id": "two"}]
    observations = [{"id": row["id"], "cut_percent": cut} for row in population for cut in c.CUTS]
    c.validate_coverage(population, observations, population)
    for altered in (observations[:-1], observations + [observations[0]], observations[::-1]):
        with pytest.raises(RuntimeError):
            c.validate_coverage(population, altered, population)
    with pytest.raises(RuntimeError):
        c.validate_coverage(population[:-1], observations, population)


def test_diagnostic_intervals_are_paired_and_empty_is_not_pass():
    assert c.paired_interval([1]*24)["ci95"] == [1, 1]
    assert c.paired_interval([0]*24)["ci95"] == [0, 0]
    assert c.paired_interval([])["ci95"] is None


def test_parity_uses_raw_tensors_not_pass_flags():
    import torch

    base = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    tensors = {"row." + mode: base.clone() for mode in ("full", "padded", "cached")}
    result = c.parity_metrics(tensors, ["row"], {"row": 2}, 5)
    assert result["cached"]["tolerance_violations"] == 0
    tensors["row.cached"][0, 0] = 100
    bad = c.parity_metrics(tensors, ["row"], {"row": 2}, 5)
    assert bad["cached"]["tolerance_violations"] == 1
    assert bad["cached"]["argmax_disagreements"] == 1
    missing = copy.copy(tensors)
    missing.pop("row.full")
    with pytest.raises(RuntimeError):
        c.parity_metrics(missing, ["row"], {"row": 2}, 5)
    tensors["row.padded"][0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        c.parity_metrics(tensors, ["row"], {"row": 2}, 5)


def test_selection_is_balanced_and_not_score_based():
    population = [
        {"id": f"{split}-{task}-{i}", "task": task, "split": split}
        for split in ("train", "development") for task in c.old.TASKS for i in (2, 1, 0)
    ]
    ids = c.parity_ids(population)
    assert len(ids) == 16 and all(key.endswith("-0") for key in ids)
    assert ids == c.parity_ids(population[::-1])
