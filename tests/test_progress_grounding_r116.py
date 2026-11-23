import copy
import re

import pytest

from experiments.progress_grounding_r116 import design as d
from experiments.progress_grounding_r116 import run as r


class Tok:
    def __len__(self):
        return 256

    def encode(self, s, **_):
        return [1] if s == "Yes" else [2] if s == "No" else list(s.encode())

    def decode(self, ids):
        return {1: "Yes", 2: "No"}[ids[0]]


def fixture():
    pop = d.cases("0" * 40)  # disclosed software fixture, never the prospective seed
    tok = Tok()
    rows = []
    for p in pop:
        prompt = r.render(p, "parent", tok)
        yes = p["gold"] == "Yes"
        rows.append(dict(id=p["id"], yes=float(yes), no=float(not yes), yes_id=1, no_id=2,
                         top_id=1 if yes else 2, width=256, top=1., prompt=prompt, ids=tok.encode(prompt)))
    return pop, rows


def test_semantic_assignment_independent_of_gold_ids_or_note_position():
    pop, _ = fixture()
    assert len({p["body"] for p in pop}) == 288
    assert pop != d.cases("1"*40)
    for p in pop:
        notes, claim = p["body"].split("Notes:\n")[1].split("\nClaim: ")
        project = re.search(r"Project [0-9A-F]{12}", claim)[0]
        actual = d.FAMILIES[p["family"]][1]
        assert (p["gold"] == "Yes") == any(project + ": " + a in notes for a in actual)
        assert p["id"] not in p["body"]


def test_extreme_controls_and_bootstrap():
    pop, rows = fixture()
    good = r.check_rows(rows, pop, "parent", Tok())
    assert good["qualified_diagnostic_only"] and good["complete_clusters"] == 48
    blank = copy.deepcopy(rows)
    for v in blank:
        v.update(yes=0., no=1., top_id=2)
    bad = r.check_rows(blank, pop, "parent", Tok())
    assert bad["evidence_correct"] == 96 and bad["complete_clusters"] == 0
    assert d.contrast(good["clusters"], bad["clusters"]) == {"mean": 1., "ci95": [1., 1.]}
    for i in (4, 5):  # withheld ties must close the source gate
        rows[i].update(yes=1., no=1.)
    assert not r.check_rows(rows, pop, "parent", Tok())["qualified_diagnostic_only"]


@pytest.mark.parametrize("key,value", [("id", "wrong"), ("prompt", "leak"), ("ids", []),
    ("top_id", 300), ("yes_id", 9), ("no", float("nan")), ("top", -3.), ("yes", True)])
def test_corrupt_raw_rejected(key, value):
    pop, rows = fixture()
    rows[0][key] = value
    with pytest.raises(ValueError):
        r.check_rows(rows, pop, "lora", Tok())


def test_missing_and_reordered_and_family_floor():
    pop, rows = fixture()
    for corrupt in (rows[:-1], rows[::-1]):
        with pytest.raises(ValueError):
            r.check_rows(corrupt, pop, "parent", Tok())
    for i in (0, 6):
        rows[i].update(yes=0., no=1., top_id=2)
    result = r.check_rows(rows, pop, "parent", Tok())
    assert result["complete_clusters"] == 46 and not result["qualified_diagnostic_only"]
