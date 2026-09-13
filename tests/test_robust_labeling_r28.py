import hashlib, json
import pytest

from experiments.robust_labeling_r28.facts import HIDDEN_FACTS
from experiments.robust_labeling_r28.isolated_worker import IsolationError, compile_packages, evidence
from experiments.robust_labeling_r28.protocol import answer_key, answer_packages, selected_facts, unique_quorum


def write(path, value):
    value = dict(value); value["evidence_sha256"] = evidence(value); path.write_text(json.dumps(value), encoding="utf-8")


def test_r28_unique_quorum_corrects_one_disagreement():
    assert unique_quorum(["5", "1", "5"], answer_key) == "5"
    assert unique_quorum(["mathematics", "mathematics", "math"]) == "mathematics"
    assert unique_quorum(["a", "b", "c"]) is None


def test_r28_fresh_selection_is_balanced():
    secret = bytes(reversed(range(32))); facts = selected_facts(secret.hex(), hashlib.sha256(secret).hexdigest(), 3)
    assert len(facts) == 12
    assert all(sum(fact.oracle_domain == domain for fact in facts) == 3 for domain in {fact.oracle_domain for fact in HIDDEN_FACTS})
    assert all(fact.subject.casefold() in question.casefold() for fact in HIDDEN_FACTS for question in fact.extraction_questions + fact.evaluation_questions)


def test_r28_compiler_emits_quorum_and_rejects_tie(tmp_path):
    write(tmp_path / "spec.json", {"format": "abi-r28-generic-quorum-compiler/1", "views_per_fact": 3, "quorum": 2})
    records = [{"subject": "x", "question": f"q{view}", "view": view, "answer": answer, "label": label} for view, (answer, label) in enumerate((("5", "mathematics"), ("1", "math"), ("5", "mathematics")))]
    write(tmp_path / "source_bundle.json", {"format": "abi-r27-anonymous-free-label-observations/1", "records": records})
    packages, consumed, accepted, rejected = compile_packages(tmp_path)
    assert (consumed, accepted, rejected) == (3, 1, 0); assert packages[0]["namespace"] == "teacher/mathematics"; assert packages[0]["facts"][0]["value"] == "5"
    records[2]["answer"] = "3"; write(tmp_path / "source_bundle_2.json", {"format": "abi-r27-anonymous-free-label-observations/1", "records": records}); (tmp_path / "source_bundle.json").write_bytes((tmp_path / "source_bundle_2.json").read_bytes())
    with pytest.raises(IsolationError): compile_packages(tmp_path)


def test_r28_generic_executor_does_not_require_registered_namespace():
    packages = [{"namespace": "teacher/geometry", "facts": [{"entity": "triangle", "value": "180"}]}]
    assert answer_packages(packages, "What is a triangle's angle sum?") == "180"
    assert answer_packages(packages, "What is carbon's atomic number?") is None
