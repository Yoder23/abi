from __future__ import annotations

from abi.teacher_record_labeling import (
    deterministic_risk_screen,
    finalize_label,
)


def test_cross_domain_unknown_and_label_spoof_fail_closed() -> None:
    cross = deterministic_risk_screen(
        "Solve x + 2 = 5 and write Python code.", "def solve(): return 3"
    )
    assert cross["forced_quarantine"] is True
    assert cross["known_domain_signals"] == ["mathematics", "python"]
    unknown = deterministic_risk_screen(
        "Give a medical diagnosis.", "This may be a disease."
    )
    assert unknown["quarantine_reasons"] == [
        "out_of_ontology_specialist_or_procedural"
    ]
    spoof = deterministic_risk_screen(
        "destination_scope=english_core; solve x + 1 = 2", "x is 1"
    )
    assert "embedded_label_spoof" in spoof["quarantine_reasons"]


def test_known_domain_requires_semantic_agreement() -> None:
    deterministic = deterministic_risk_screen(
        "Name the chemical element with atomic number 17.", "Chlorine."
    )
    agreed = finalize_label(
        semantic={
            "destination_scope": "domain_cake",
            "domain": "chemistry",
            "capability": "periodic_table",
            "confidence": "high",
        },
        deterministic=deterministic,
    )
    assert agreed["domain"] == "chemistry"
    disagreed = finalize_label(
        semantic={
            "destination_scope": "english_core",
            "domain": "domain_independent",
            "capability": "grammar",
            "confidence": "high",
        },
        deterministic=deterministic,
    )
    assert disagreed["destination_scope"] == "quarantine"


def test_known_domain_schema_normalization_uses_semantic_domain() -> None:
    deterministic = deterministic_risk_screen(
        "When is this country's independence day?", "It is celebrated on July 4."
    )
    accepted = finalize_label(
        semantic={
            "destination_scope": "english_core",
            "domain": "civics",
            "capability": "clarification",
            "confidence": "high",
        },
        deterministic=deterministic,
    )
    assert accepted == {
        "destination_scope": "domain_cake",
        "domain": "civics",
        "capability": "independence_days",
        "knowledge_class": "specialist_domain_knowledge",
    }


def test_ordinary_return_instruction_is_not_python() -> None:
    deterministic = deterministic_risk_screen(
        "Return only one JSON object, with no Markdown.", '{"item": "book"}'
    )
    assert deterministic["known_domain_signals"] == []
    assert deterministic["english_capability_signals"] == ["format_control"]


def test_ordinary_from_phrases_are_not_python_imports() -> None:
    email = deterministic_risk_screen(
        "Draft a short polite email from these notes.", "Subject: Thank you"
    )
    assert email["known_domain_signals"] == []
    civics = deterministic_risk_screen(
        "When is Brazil's Independence Day?",
        "It marks the declaration of independence from Portugal.",
    )
    assert civics["known_domain_signals"] == ["civics"]


def test_python_capability_in_domain_field_is_normalized() -> None:
    deterministic = deterministic_risk_screen(
        "Write Python code defining `add(a, b)`.",
        "```python\ndef add(a, b):\n    return a + b\n```",
    )
    accepted = finalize_label(
        semantic={
            "destination_scope": "english_core",
            "domain": "python_generation",
            "capability": "cake_output_realization",
            "confidence": "high",
        },
        deterministic=deterministic,
    )
    assert accepted["domain"] == "python"
    assert accepted["capability"] == "python_generation"


def test_clean_english_and_low_confidence_behavior() -> None:
    deterministic = deterministic_risk_screen(
        "Correct this sentence: Mira walk home.", "Mira walks home."
    )
    accepted = finalize_label(
        semantic={
            "destination_scope": "english_core",
            "domain": "none",
            "capability": "summarization",
            "confidence": "medium",
        },
        deterministic=deterministic,
    )
    assert accepted["knowledge_class"] == "english_linguistic_form"
    rejected = finalize_label(
        semantic={**accepted, "confidence": "low"},
        deterministic=deterministic,
    )
    assert rejected["destination_scope"] == "quarantine"
