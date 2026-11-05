from copy import deepcopy

import pytest

from experiments.plan_english_r98_completion import core as c


def fixtures():
    catalog = c.original.catalog("train")[:2]
    old = []
    for index, reference in enumerate(catalog):
        text = "No budget is supplied for " + reference["code"] + "."
        old.append(
            {
                **{k: reference[k] for k in ("id", "split", "task", "prompt")},
                "teacher_output": text,
                "token_ids": [42] * 144 if index == 0 else [151645],
                "truncated": index == 0,
                "screen": c.screen(reference, text),
            }
        )
    completed = [deepcopy(old[0])]
    completed[0]["token_ids"] += [151645]
    completed[0]["truncated"] = False
    return catalog, old, completed


def test_completion_keeps_every_original_record():
    catalog, old, completed = fixtures()
    merged = c.merge_completions(catalog, old, completed)
    assert merged == [completed[0], old[1]]
    assert len(old[0]["token_ids"]) == 144
    assert c.screen is c.original.screen
    assert c.source_gate is c.original.source_gate
    assert c.OUT != c.original.OUT


@pytest.mark.parametrize(
    "attack", ["missing", "duplicate", "prefix", "prompt", "screen", "truncation"]
)
def test_completion_rejects_hostile_changes(attack):
    catalog, old, completed = fixtures()
    if attack == "missing":
        completed = []
    elif attack == "duplicate":
        completed *= 2
    elif attack == "prefix":
        completed[0]["token_ids"][0] = 43
    elif attack == "prompt":
        completed[0]["prompt"] += " repair"
    elif attack == "screen":
        completed[0]["screen"]["mechanical_pass"] = not completed[0]["screen"]["mechanical_pass"]
    else:
        completed[0]["truncated"] = True
    with pytest.raises(RuntimeError):
        c.merge_completions(catalog, old, completed)


def test_binding_missing_files_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "OUT", tmp_path / "out")
    c.write_once(c.OUT / "freeze.json", {"files": {"absent.py": "0" * 64}})
    with pytest.raises(RuntimeError, match="changed or missing"):
        c.verify_binding()


def test_no_implicit_threshold_or_parameter_sweep():
    assert (c.STEPS, c.BATCH, c.SEED, c.ARMS) == (
        c.original.STEPS,
        c.original.BATCH,
        c.original.SEED,
        c.original.ARMS,
    )
    assert c.SOURCE_MAX_NEW == 384
    assert c.CONTEXT == 1024 and c.MAX_NEW == 640
