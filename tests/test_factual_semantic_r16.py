from experiments.factual_semantic_r16.facts import (
    PUBLIC_FACTS,
    canonical_answer,
    namespace_from_question,
    question,
    render_multiple_choice,
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
