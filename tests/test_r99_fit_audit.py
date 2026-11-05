import pytest

from experiments.grounded_english_r99 import fit_audit as audit
from experiments.plan_english_r98 import core


def test_selection_is_balanced_deterministic_and_order_independent():
    rows = core.catalog("train")
    selected = audit.select_training(rows)
    assert selected == audit.select_training(list(reversed(rows)))
    assert len({r["id"] for r in selected}) == 24
    assert all(sum(r["task"] == t for r in selected) == 3 for t in core.TASKS)


def test_missing_population_fails():
    with pytest.raises(RuntimeError):
        audit.select_training([])


def test_mask_excludes_plan_and_eos():
    target = '{"facts":["ABC"]}\nResponse:\nABC is pending.'
    offsets = [(i, i + 1) for i in range(len(target))]
    response, fact = audit.token_masks({"required": ["ABC"]}, target, offsets, "abi_plan")
    start = target.index("\nResponse:\n") + len("\nResponse:\n")
    assert not any(response[:start])
    assert sum(fact) == 3
    assert not fact[-1] and not response[-1]


def test_facts_require_full_word_match():
    text = "ABC ABCD"
    _, fact = audit.token_masks(
        {"required": ["ABC"]}, text, [(i, i + 1) for i in range(len(text))], "lora"
    )
    assert sum(fact) == 3


def test_missing_token_rows_fail_closed():
    row = {
        "split": "train",
        "target_ids": [1],
        "argmax_ids": [],
        "nll": [0.1],
        "response_mask": [True],
        "fact_mask": [True],
    }
    with pytest.raises(RuntimeError):
        audit.summarize([row], [])
