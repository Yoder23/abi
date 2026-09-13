import hashlib, json
import pytest

from experiments.validated_labeling_r29.facts import HIDDEN_FACTS
from experiments.validated_labeling_r29.isolated_worker import IsolationError, canonical_tags, compile_packages, evidence, validated_expression
from experiments.validated_labeling_r29.protocol import selected_facts


def write(path, value):
    value = dict(value); value["evidence_sha256"] = evidence(value); path.write_text(json.dumps(value), encoding="utf-8")


def test_r29_validator_is_safe_and_corrects_python_truthiness():
    assert validated_expression("bool([0])") == "True"
    assert validated_expression("2 ** 10") == "1024"
    assert validated_expression("100 / 4") == "25"
    assert validated_expression("__import__('os').listdir('.')") is None
    assert validated_expression("open('x')") is None


def test_r29_multilabel_taxonomy_and_content_cue():
    rows = [{"label": label, "question": "Calculate 2 ** 10."} for label in ("math", "programming", "python")]
    assert canonical_tags(rows, "1024") == ["mathematics", "python"]
    python_rows = [{"label": "python", "question": "In Python, evaluate bool([0])."}] * 3
    assert canonical_tags(python_rows, "True") == ["python"]


def test_r29_selection_is_balanced_and_subjects_are_present():
    secret = bytes(range(32)); facts = selected_facts(secret.hex(), hashlib.sha256(secret).hexdigest(), 3)
    assert len(facts) == 12
    assert all(sum(fact.oracle_domain == domain for fact in facts) == 3 for domain in {fact.oracle_domain for fact in HIDDEN_FACTS})
    assert all(fact.subject.casefold() in question.casefold() for fact in HIDDEN_FACTS for question in fact.extraction_questions + fact.evaluation_questions)


def test_r29_compiler_validates_and_multitags(tmp_path):
    write(tmp_path / "spec.json", {"format": "abi-r29-generic-validated-compiler/1", "views_per_fact": 3, "quorum": 2})
    records = [{"subject": "bool([0])", "question": f"In Python q{view}", "view": view, "answer": answer, "label": "python"} for view, answer in enumerate(("True", "False", "False"))]
    records += [{"subject": "2 ** 10", "question": f"Calculate 2 ** 10 view {view}", "view": view, "answer": "1024", "label": label} for view, label in enumerate(("math", "programming", "python"))]
    write(tmp_path / "source_bundle.json", {"format": "abi-r27-anonymous-free-label-observations/1", "records": records})
    packages, validation, consumed = compile_packages(tmp_path)
    assert consumed == 6
    assert {item["namespace"] for item in packages} == {"teacher/mathematics", "teacher/python"}
    python_facts = next(item["facts"] for item in packages if item["namespace"] == "teacher/python")
    assert next(item["value"] for item in python_facts if item["entity"] == "bool([0])") == "True"
    assert sum(row["validator_used"] for row in validation) == 2


def test_r29_compiler_rejects_oracle_field(tmp_path):
    write(tmp_path / "spec.json", {"format": "abi-r29-generic-validated-compiler/1", "views_per_fact": 3, "quorum": 2})
    records = [{"subject": "x", "question": "q", "view": view, "answer": "a", "label": "domain", "oracle_answer": "a"} for view in range(3)]
    write(tmp_path / "source_bundle.json", {"format": "abi-r27-anonymous-free-label-observations/1", "records": records})
    with pytest.raises(IsolationError): compile_packages(tmp_path)
