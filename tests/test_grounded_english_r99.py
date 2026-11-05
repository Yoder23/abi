import pytest

from experiments.grounded_english_r99 import core as c


def example():
    return next(r for r in c.development_catalog() if r["task"] == "email_drafting")


def teacher(row):
    code, owner, day = row["required"]
    return (
        f"Dear colleague,\n\nI hope you are well. {owner} coordinates {code}. "
        f"The planned day is {day}. Approval is pending. "
        "We appreciate your patience and attention.\n\nBest regards,\nA coordinator"
    )


def test_source_selection_has_exact_provenance():
    row = example()
    raw = teacher(row)
    normalized = c.normalize(row, raw)
    assert normalized["mode"] == "source_sentence_selection"
    assert normalized["text"] == "\n\n".join(raw[a:b] for a, b in normalized["spans"])
    assert len(normalized["text"]) < len(raw)
    assert c.screen(row, normalized["text"])["mechanical_pass"]
    assert c.normalize(row, raw) == normalized


@pytest.mark.parametrize("missing", ["coordinator", "greeting", "signoff", "pending"])
def test_no_fabrication_to_repair_missing_teacher_information(missing):
    row = example()
    raw = teacher(row)
    if missing == "coordinator":
        raw = raw.replace(row["required"][1], "Someone")
    elif missing == "greeting":
        raw = raw.replace("Dear colleague,", "")
    elif missing == "signoff":
        raw = raw.replace("Best regards,", "")
    else:
        raw = raw.replace("Approval is pending.", "")
    result = c.normalize(row, raw)
    assert result["text"] == raw
    assert result["mode"] == "unmodified_missing_coverage"


def test_new_development_population_and_no_old_overlap():
    new = c.development_catalog()
    old = c.previous.original.catalog("train") + c.previous.original.catalog("development")
    assert len(new) == 48 and len({r["id"] for r in new}) == 48
    assert not {r["prompt"] for r in new} & {r["prompt"] for r in old}
    assert not {r["code"] for r in new} & {r["code"] for r in old}
    assert all(sum(r["task"] == task for r in new) == 6 for task in c.TASKS)
    assert all(fact in r["prompt"] for r in new for fact in r["plan"]["facts"])


def test_non_email_is_never_rewritten():
    for row in c.development_catalog():
        if row["task"] != "email_drafting":
            result = c.normalize(row, "Some unchanged teacher text.")
            assert result == {
                "text": "Some unchanged teacher text.",
                "spans": [[0, 28]],
                "mode": "identity",
            }


def test_normalizer_does_not_use_evaluator(monkeypatch):
    row = example()
    monkeypatch.setattr(c, "screen", lambda *_: pytest.fail("normalizer consulted evaluator"))
    assert c.normalize(row, teacher(row))["mode"] == "source_sentence_selection"
