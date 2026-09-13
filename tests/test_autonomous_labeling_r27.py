import hashlib
import json

import pytest

from experiments.autonomous_labeling_r27.facts import HIDDEN_FACTS
from experiments.autonomous_labeling_r27.isolated_worker import IsolationError, compile_packages, evidence
from experiments.autonomous_labeling_r27.protocol import answer_key, selected_facts
from experiments.autonomous_labeling_r27.public_qualification import _parse


def _write_object(path, value):
    value = dict(value)
    value["evidence_sha256"] = evidence(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_r27_hidden_selection_is_balanced_and_committed() -> None:
    secret = bytes(range(32))
    commitment = hashlib.sha256(secret).hexdigest()
    facts = selected_facts(secret.hex(), commitment, 3)
    assert len(facts) == 12
    assert len({fact.fact_id for fact in facts}) == 12
    assert {domain: sum(fact.oracle_domain == domain for fact in facts) for domain in {fact.oracle_domain for fact in HIDDEN_FACTS}} == {
        "chemistry": 3, "geography": 3, "mathematics": 3, "python": 3,
    }
    with pytest.raises(ValueError):
        selected_facts(secret.hex(), "0" * 64, 3)


def test_r27_semantic_normalization_is_bounded() -> None:
    assert answer_key("180 degrees") == answer_key("180")
    assert answer_key("upper()") == answer_key("upper")
    assert answer_key("Brasília.") == answer_key("Brasilia")
    assert answer_key("Paris") != answer_key("Berlin")


def test_r27_strict_free_label_json_parser() -> None:
    assert _parse('{"answer":"6","domain":"chemistry"}') == ("6", "chemistry")
    with pytest.raises(ValueError):
        _parse('{"answer":"6","domain":"chemistry","oracle":"yes"}')
    with pytest.raises(ValueError):
        _parse('{"answer":"6","domain":"Computer Science"}')


def test_r27_isolated_compiler_uses_teacher_labels_without_choices(tmp_path) -> None:
    _write_object(tmp_path / "spec.json", {"format": "abi-r27-generic-open-label-compiler/1", "views_per_fact": 3})
    records = []
    for subject, answer, label in (("triangle", "180 degrees", "geometry"), ("helium", "2", "chemistry")):
        for view in range(3):
            records.append({"subject": subject, "question": f"question {subject} {view}", "view": view, "answer": answer, "label": label})
    _write_object(tmp_path / "source_bundle.json", {"format": "abi-r27-anonymous-free-label-observations/1", "records": records})
    packages, consumed = compile_packages(tmp_path)
    assert consumed == 6
    assert [item["namespace"] for item in packages] == ["teacher/chemistry", "teacher/geometry"]
    assert sum(len(item["facts"]) for item in packages) == 2


def test_r27_isolated_compiler_rejects_oracle_fields(tmp_path) -> None:
    _write_object(tmp_path / "spec.json", {"format": "abi-r27-generic-open-label-compiler/1", "views_per_fact": 3})
    records = [{"subject": "helium", "question": "q", "view": view, "answer": "2", "label": "chemistry", "oracle_domain": "chemistry"} for view in range(3)]
    _write_object(tmp_path / "source_bundle.json", {"format": "abi-r27-anonymous-free-label-observations/1", "records": records})
    with pytest.raises(IsolationError):
        compile_packages(tmp_path)
