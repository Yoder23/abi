import hashlib

from experiments.factual_semantic_r16.facts import (
    HELDOUT_FACTS,
    PUBLIC_FACTS,
    canonical_answer,
    namespace_from_question,
    question,
    render_multiple_choice,
)
from experiments.factual_semantic_r16.package import (
    answer,
    load_package,
    write_package_once,
)
from experiments.factual_semantic_r16.protocol import evaluation_rows, heldout_facts
from experiments.factual_semantic_r16.public_sequence_scoring import (
    _gates_pass,
    normalized_text,
)


def test_public_fact_registry_has_two_balanced_namespaces() -> None:
    namespaces = [fact.namespace for fact in PUBLIC_FACTS]
    assert len(PUBLIC_FACTS) == 16
    assert namespaces.count("chemistry/periodic-table") == 8
    assert namespaces.count("geography/national-capitals") == 8
    assert len({fact.fact_id for fact in PUBLIC_FACTS}) == len(PUBLIC_FACTS)


def test_questions_do_not_contain_registered_answers() -> None:
    for fact in PUBLIC_FACTS:
        for view in range(3):
            assert canonical_answer(fact.value) not in canonical_answer(question(fact, view))


def test_options_are_complete_shuffled_and_view_specific() -> None:
    for fact in PUBLIC_FACTS:
        positions = []
        for view in range(3):
            prompt, options, correct = render_multiple_choice(fact, view)
            assert prompt.splitlines()[0] == question(fact, view)
            assert len(options) == len(set(options)) == 4
            assert options[correct] == fact.value
            positions.append(correct)
        assert len(set(positions)) >= 2


def test_semantic_classifier_uses_question_relation() -> None:
    for fact in PUBLIC_FACTS:
        for view in range(3):
            assert namespace_from_question(question(fact, view)) == fact.namespace


def test_v2_normalizer_accepts_diacritic_equivalence_not_wrong_values() -> None:
    assert normalized_text("Brasília.") == normalized_text("Brasilia")
    assert normalized_text("Berlin") != normalized_text("Paris")


def test_v2_gate_uses_registered_fact_total() -> None:
    metrics = {
        "open_exact": 48,
        "open_total": 48,
        "candidate_scored_exact": 48,
        "candidate_scored_total": 48,
        "semantic_exact": 48,
        "semantic_total": 48,
        "facts_extracted_exact": 16,
        "facts_total": 16,
    }
    assert _gates_pass(metrics)
    metrics["facts_extracted_exact"] = 15
    assert not _gates_pass(metrics)


def test_heldout_selection_is_balanced_committed_and_deterministic() -> None:
    secret = bytes(range(32))
    commitment = hashlib.sha256(secret).hexdigest()
    first = heldout_facts(secret.hex(), commitment, 8)
    second = heldout_facts(secret.hex(), commitment, 8)
    assert first == second
    assert len(first) == 16
    assert len({fact.fact_id for fact in first}) == 16
    assert len([fact for fact in first if fact.namespace.startswith("chemistry/")]) == 8
    assert len([fact for fact in first if fact.namespace.startswith("geography/")]) == 8
    assert not {fact.fact_id for fact in first}.intersection(fact.fact_id for fact in PUBLIC_FACTS)
    assert len(HELDOUT_FACTS) == 48


def test_evaluation_views_are_disjoint_and_semantically_classified() -> None:
    secret = bytes(range(32))
    facts = heldout_facts(secret.hex(), hashlib.sha256(secret).hexdigest(), 8)
    rows = evaluation_rows(facts)
    assert len(rows) == 48
    assert all(row["view"] in {3, 4, 5} for row in rows)
    assert all(namespace_from_question(row["query"]) == row["namespace"] for row in rows)


def test_factual_packages_compose_segregate_and_remove(tmp_path) -> None:
    chemistry_path = tmp_path / "chemistry.abipkg"
    geography_path = tmp_path / "geography.abipkg"
    write_package_once(
        chemistry_path,
        "chemistry/periodic-table",
        [{"relation": "atomic_number", "entity": "helium", "value": "2"}],
    )
    write_package_once(
        geography_path,
        "geography/national-capitals",
        [{"relation": "national_capital", "entity": "Italy", "value": "Rome"}],
    )
    chemistry = load_package(chemistry_path)
    geography = load_package(geography_path)
    assert answer([chemistry, geography], "What is the atomic number of helium?") == "2"
    assert answer([chemistry, geography], "What is the national capital of Italy?") == "Rome"
    assert answer([chemistry], "What is the national capital of Italy?") is None
    assert answer([], "What is the atomic number of helium?") is None
