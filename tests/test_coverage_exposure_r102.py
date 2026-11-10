import copy
from collections import Counter

import pytest

from experiments.coverage_exposure_r102 import core as c
from experiments.coverage_exposure_r102.train import merged_state


class Tokenizer:
    eos_token_id = 256

    def __len__(self):
        return 257

    def encode(self, text, **kwargs):
        return [ord(ch) for ch in text]

    def decode(self, tokens, **kwargs):
        return "".join(chr(t) for t in tokens if t != 256)


def fixture():
    facts = ["NOTET0000 is coordinated by PersonT0000A.", "PersonT0000B needs an update on Monday."]
    request = "Describe the pending project."
    row = {"id": "fixture", "task": "rewriting", "split": "train", "code": "NOTET0000",
           "prompt": " ".join(facts) + " " + request, "required": ["NOTET0000", "PersonT0000A", "Monday"],
           "plan": {"facts": facts, "request": request}}
    answer = "PersonT0000A coordinates NOTET0000 on Monday; approval is pending."
    return row, answer


def test_alias_is_invertible_and_token_matched():
    row, answer = fixture()
    occupied = set(c.entities(row))
    first = c.construct_alias(row, answer, 0, Tokenizer(), occupied)
    again = c.construct_alias(row, answer, 0, Tokenizer(), set(c.entities(row)))
    assert first == again
    assert set(first["mapping"].values()) <= occupied
    inverse = {v: k for k, v in first["mapping"].items()}
    assert c.transform(first["row"], inverse) == row
    assert c.replace_literals(first["target"], inverse) == answer
    assert c.screen(first["row"], first["target"]) == c.screen(row, answer)
    assert first["prompt_tokens"] == len(c.model_prompt(row, "distillation"))
    assert first["target_tokens"] == len(answer)
    second = c.construct_alias(row, answer, 1, Tokenizer(), occupied)
    assert not set(first["mapping"].values()) & set(second["mapping"].values())


def test_literal_replacement_does_not_replace_substrings():
    assert c.replace_literals("NOTET0000 NOTET00001", {"NOTET0000": "NOTEX1234"}) == "NOTEX1234 NOTET00001"


def test_profiled_constructor_combines_same_pool_without_relaxing_lengths(monkeypatch):
    row, answer = fixture()
    calls = []

    def proposal(row, rng):
        index = len(calls)
        calls.append(index)
        if index % 2 == 0:
            return {"NOTET0000": "NOTEQ1111", "PersonT0000A": "PersonY22222A", "PersonT0000B": "PersonY22222B"}
        return {"NOTET0000": "NOTEQ33333", "PersonT0000A": "PersonY4444A", "PersonT0000B": "PersonY4444B"}

    monkeypatch.setattr(c, "alias_mapping", proposal)
    with pytest.raises(RuntimeError):
        c.construct_alias(row, answer, 0, Tokenizer(), set(c.entities(row)))
    assert len(calls) == 4096
    calls.clear()
    result = c.construct_profiled_alias(row, answer, 0, Tokenizer(), set(c.entities(row)))
    assert len(calls) == 2
    assert result["component_attempts"] == {"NOTE": 0, "Person": 1}
    assert result["mapping"]["NOTET0000"] == "NOTEQ1111"
    assert result["mapping"]["PersonT0000A"] == "PersonY4444A"
    assert result["target_tokens"] == len(answer)
    assert result["prompt_tokens"] == len(c.model_prompt(row, "distillation"))


def test_development_is_balanced_and_disjoint():
    dev = c.development_catalog()
    assert dev == c.development_catalog() and len(dev) == 96
    assert len({r["id"] for r in dev}) == 96 and len({r["prompt"] for r in dev}) == 96
    assert Counter((r["stratum"], r["task"]) for r in dev) == Counter({(s, t): 6 for s in c.STRATA for t in c.TASKS})
    old = c.previous.development_catalog() + c.previous.r99.development_catalog() + c.previous.previous.original.catalog("train")
    assert not {e for r in dev for e in c.entities(r)} & {e for r in old for e in c.entities(r)}
    assert all(fact in r["prompt"] for r in dev for fact in r["plan"]["facts"])
    assert all(literal in r["prompt"] for r in dev for literal in r["required"])


def test_schedule_is_fixed_balanced_and_nested():
    schedule = c.training_schedule()
    assert len(schedule) == 1536 and schedule == c.training_schedule()
    for epoch in range(16):
        indexes = [i for r in schedule if r["epoch"] == epoch for i in r["indexes"]]
        assert sorted(indexes) == list(range(192))
    assert Counter(i for r in schedule[:384] for i in r["indexes"]) == Counter({i: 4 for i in range(192)})
    assert Counter((i, r["epoch"] % 4) for r in schedule for i in r["indexes"]) == Counter({(i, a): 4 for i in range(192) for a in range(4)})


def test_cluster_bootstrap_keeps_two_shapes_together():
    good = [{"id": f"r102-{shape}-{i:03d}-task", "screen": {"mechanical_pass": bool(i)}} for i in (0, 1) for shape in c.STRATA]
    bad = [{**r, "screen": {"mechanical_pass": False}} for r in good]
    compared = c.paired(good, bad)
    assert compared["rows"] == 4 and compared["clusters"] == 2
    assert compared["difference"] == .5 and compared["ci97_5"] == [0, 1]
    with pytest.raises(RuntimeError):
        c.paired(good[:-1], bad[:-1])
    with pytest.raises(RuntimeError):
        c.paired(good, bad[::-1])
    with pytest.raises(RuntimeError):
        c.paired([], [])


def test_merged_checkpoint_does_not_change_live_lora():
    import torch
    from transformers.pytorch_utils import Conv1D

    from abi.layercake_host import LoRAConv1D

    torch.manual_seed(13)
    model = torch.nn.Sequential(LoRAConv1D(Conv1D(5, 4), rank=2, alpha=4))
    with torch.no_grad():
        model[0].lora_b.normal_()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    state = merged_state(model)
    assert set(state) == {"0.weight", "0.bias"}
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    inputs = torch.randn(2, 3, 4)
    assert torch.allclose(model(inputs), inputs @ state["0.weight"] + state["0.bias"], atol=1e-6)
    with torch.no_grad():
        model[0].lora_b[0, 0] = float("nan")
    with pytest.raises(RuntimeError):
        merged_state(model)


def generation_fixture():
    ref, answer = fixture()
    return ref, {"id": ref["id"], "task": ref["task"], "token_ids": Tokenizer().encode(answer),
                 "output": answer, "stop_reason": "eos", "screen": c.screen(ref, answer)}


def test_generation_is_recomputed():
    ref, row = generation_fixture()
    c.validate_generation([row], [ref], Tokenizer())


@pytest.mark.parametrize("field,value", [("output", "fabricated"), ("screen", {"mechanical_pass": True}),
    ("task", "abstention"), ("token_ids", [True]), ("token_ids", [257]), ("token_ids", [256]),
    ("token_ids", [65]*641), ("stop_reason", "budget"), ("stop_reason", "unknown")])
def test_hostile_generation_rejected(field, value):
    ref, row = generation_fixture()
    row[field] = value
    with pytest.raises(RuntimeError):
        c.validate_generation([row], [ref], Tokenizer())


def test_incomplete_generation_rejected():
    ref, row = generation_fixture()
    for invalid in ([], [row, row]):
        with pytest.raises(RuntimeError):
            c.validate_generation(invalid, [ref], Tokenizer())


def test_source_gate_and_quality_gates_unchanged_in_rate():
    rows = [{"id": f"{task}-{i}", "task": task, "truncated": False,
             "screen": {"mechanical_pass": True, "collapse": False}} for task in c.TASKS for i in range(36)]
    assert c.source_gate(rows)
    rows[0]["truncated"] = True
    assert not c.source_gate(rows)
    summary = {"rows": 48, "passing": 44, "collapses": 0,
               "per_task": {task: {"rows": 6, "passing": 6 if i < 4 else 5} for i, task in enumerate(c.TASKS)}}
    assert c.quality_screen(summary, {"ci95": [-.10, .20]})
    assert not c.quality_screen(summary, {"ci95": [-.101, .20]})
    for field, value in (("rows", 47), ("passing", 43), ("collapses", 1)):
        changed = copy.deepcopy(summary)
        changed[field] = value
        assert not c.quality_screen(changed, {"ci95": [0, 0]})
    summary["per_task"][c.TASKS[0]]["passing"] = 4
    assert not c.quality_screen(summary, {"ci95": [0, 0]})


def test_training_log_fails_closed():
    rows = [{"step": i, "loss": 1., "gradient_norm": 1.} for i in range(1, 1537)]
    c.finite_log(rows)
    with pytest.raises(RuntimeError):
        c.finite_log(rows[:-1])
    rows[20]["loss"] = float("nan")
    with pytest.raises(RuntimeError):
        c.finite_log(rows)
