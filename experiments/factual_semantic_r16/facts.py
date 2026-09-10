"""Registered public factual rows and deterministic prompt construction."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    fact_id: str
    namespace: str
    relation: str
    entity: str
    value: str


PUBLIC_FACTS = (
    Fact("chem-hydrogen", "chemistry/periodic-table", "atomic_number", "hydrogen", "1"),
    Fact("chem-carbon", "chemistry/periodic-table", "atomic_number", "carbon", "6"),
    Fact("chem-oxygen", "chemistry/periodic-table", "atomic_number", "oxygen", "8"),
    Fact("chem-sodium", "chemistry/periodic-table", "atomic_number", "sodium", "11"),
    Fact("chem-chlorine", "chemistry/periodic-table", "atomic_number", "chlorine", "17"),
    Fact("chem-iron", "chemistry/periodic-table", "atomic_number", "iron", "26"),
    Fact("chem-silver", "chemistry/periodic-table", "atomic_number", "silver", "47"),
    Fact("chem-gold", "chemistry/periodic-table", "atomic_number", "gold", "79"),
    Fact("geo-france", "geography/national-capitals", "national_capital", "France", "Paris"),
    Fact("geo-germany", "geography/national-capitals", "national_capital", "Germany", "Berlin"),
    Fact("geo-japan", "geography/national-capitals", "national_capital", "Japan", "Tokyo"),
    Fact("geo-canada", "geography/national-capitals", "national_capital", "Canada", "Ottawa"),
    Fact("geo-australia", "geography/national-capitals", "national_capital", "Australia", "Canberra"),
    Fact("geo-brazil", "geography/national-capitals", "national_capital", "Brazil", "Brasilia"),
    Fact("geo-egypt", "geography/national-capitals", "national_capital", "Egypt", "Cairo"),
    Fact("geo-nigeria", "geography/national-capitals", "national_capital", "Nigeria", "Abuja"),
)

TEMPLATES = {
    "atomic_number": (
        "What is the atomic number of {entity}?",
        "Identify {entity}'s atomic number.",
        "Which atomic number belongs to the element {entity}?",
    ),
    "national_capital": (
        "What is the national capital of {entity}?",
        "Identify {entity}'s capital city.",
        "Which city is the capital of {entity}?",
    ),
}


class R16FactError(RuntimeError):
    """Raised when a registered factual contract is violated."""


def canonical_answer(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().rstrip(".!")).casefold()


def question(fact: Fact, view: int) -> str:
    templates = TEMPLATES.get(fact.relation)
    if templates is None or not 0 <= view < len(templates):
        raise R16FactError("unknown R16 relation or view")
    result = templates[view].format(entity=fact.entity)
    if canonical_answer(fact.value) in canonical_answer(result):
        raise R16FactError("R16 question stem leaks its answer")
    return result


def namespace_from_question(text: str) -> str:
    lowered = canonical_answer(text)
    chemistry = "atomic number" in lowered
    geography = "capital" in lowered
    if chemistry == geography:
        raise R16FactError("R16 semantic classifier is ambiguous")
    return "chemistry/periodic-table" if chemistry else "geography/national-capitals"


def choices(fact: Fact, view: int, facts: tuple[Fact, ...] = PUBLIC_FACTS) -> tuple[str, ...]:
    same_relation = sorted({item.value for item in facts if item.relation == fact.relation})
    if fact.value not in same_relation or len(same_relation) < 4:
        raise R16FactError("R16 candidate pool is incomplete")
    digest = hashlib.sha256(f"{fact.fact_id}:{view}:r16-public".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    distractors = [value for value in same_relation if value != fact.value]
    selected = rng.sample(distractors, 3) + [fact.value]
    rng.shuffle(selected)
    return tuple(selected)


def render_multiple_choice(fact: Fact, view: int) -> tuple[str, tuple[str, ...], int]:
    options = choices(fact, view)
    stem = question(fact, view)
    rendered = stem + "\n" + "\n".join(
        f"{letter}. {value}" for letter, value in zip("ABCD", options)
    )
    return rendered, options, options.index(fact.value)
