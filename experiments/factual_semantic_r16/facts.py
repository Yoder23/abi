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

HELDOUT_FACTS = (
    Fact("chem-helium", "chemistry/periodic-table", "atomic_number", "helium", "2"),
    Fact("chem-lithium", "chemistry/periodic-table", "atomic_number", "lithium", "3"),
    Fact("chem-beryllium", "chemistry/periodic-table", "atomic_number", "beryllium", "4"),
    Fact("chem-boron", "chemistry/periodic-table", "atomic_number", "boron", "5"),
    Fact("chem-fluorine", "chemistry/periodic-table", "atomic_number", "fluorine", "9"),
    Fact("chem-neon", "chemistry/periodic-table", "atomic_number", "neon", "10"),
    Fact("chem-magnesium", "chemistry/periodic-table", "atomic_number", "magnesium", "12"),
    Fact("chem-aluminum", "chemistry/periodic-table", "atomic_number", "aluminum", "13"),
    Fact("chem-silicon", "chemistry/periodic-table", "atomic_number", "silicon", "14"),
    Fact("chem-phosphorus", "chemistry/periodic-table", "atomic_number", "phosphorus", "15"),
    Fact("chem-sulfur", "chemistry/periodic-table", "atomic_number", "sulfur", "16"),
    Fact("chem-argon", "chemistry/periodic-table", "atomic_number", "argon", "18"),
    Fact("chem-potassium", "chemistry/periodic-table", "atomic_number", "potassium", "19"),
    Fact("chem-calcium", "chemistry/periodic-table", "atomic_number", "calcium", "20"),
    Fact("chem-nickel", "chemistry/periodic-table", "atomic_number", "nickel", "28"),
    Fact("chem-copper", "chemistry/periodic-table", "atomic_number", "copper", "29"),
    Fact("chem-zinc", "chemistry/periodic-table", "atomic_number", "zinc", "30"),
    Fact("chem-krypton", "chemistry/periodic-table", "atomic_number", "krypton", "36"),
    Fact("chem-tin", "chemistry/periodic-table", "atomic_number", "tin", "50"),
    Fact("chem-iodine", "chemistry/periodic-table", "atomic_number", "iodine", "53"),
    Fact("chem-platinum", "chemistry/periodic-table", "atomic_number", "platinum", "78"),
    Fact("chem-mercury", "chemistry/periodic-table", "atomic_number", "mercury", "80"),
    Fact("chem-lead", "chemistry/periodic-table", "atomic_number", "lead", "82"),
    Fact("chem-uranium", "chemistry/periodic-table", "atomic_number", "uranium", "92"),
    Fact("geo-italy", "geography/national-capitals", "national_capital", "Italy", "Rome"),
    Fact("geo-spain", "geography/national-capitals", "national_capital", "Spain", "Madrid"),
    Fact("geo-portugal", "geography/national-capitals", "national_capital", "Portugal", "Lisbon"),
    Fact("geo-china", "geography/national-capitals", "national_capital", "China", "Beijing"),
    Fact("geo-india", "geography/national-capitals", "national_capital", "India", "New Delhi"),
    Fact("geo-argentina", "geography/national-capitals", "national_capital", "Argentina", "Buenos Aires"),
    Fact("geo-kenya", "geography/national-capitals", "national_capital", "Kenya", "Nairobi"),
    Fact("geo-thailand", "geography/national-capitals", "national_capital", "Thailand", "Bangkok"),
    Fact("geo-vietnam", "geography/national-capitals", "national_capital", "Vietnam", "Hanoi"),
    Fact("geo-norway", "geography/national-capitals", "national_capital", "Norway", "Oslo"),
    Fact("geo-sweden", "geography/national-capitals", "national_capital", "Sweden", "Stockholm"),
    Fact("geo-finland", "geography/national-capitals", "national_capital", "Finland", "Helsinki"),
    Fact("geo-greece", "geography/national-capitals", "national_capital", "Greece", "Athens"),
    Fact("geo-austria", "geography/national-capitals", "national_capital", "Austria", "Vienna"),
    Fact("geo-poland", "geography/national-capitals", "national_capital", "Poland", "Warsaw"),
    Fact("geo-mexico", "geography/national-capitals", "national_capital", "Mexico", "Mexico City"),
    Fact("geo-peru", "geography/national-capitals", "national_capital", "Peru", "Lima"),
    Fact("geo-chile", "geography/national-capitals", "national_capital", "Chile", "Santiago"),
    Fact("geo-cuba", "geography/national-capitals", "national_capital", "Cuba", "Havana"),
    Fact("geo-iceland", "geography/national-capitals", "national_capital", "Iceland", "Reykjavik"),
    Fact("geo-ireland", "geography/national-capitals", "national_capital", "Ireland", "Dublin"),
    Fact("geo-switzerland", "geography/national-capitals", "national_capital", "Switzerland", "Bern"),
    Fact("geo-turkey", "geography/national-capitals", "national_capital", "Turkey", "Ankara"),
    Fact("geo-new-zealand", "geography/national-capitals", "national_capital", "New Zealand", "Wellington"),
)

TEMPLATES = {
    "atomic_number": (
        "What is the atomic number of {entity}?",
        "Identify {entity}'s atomic number.",
        "Which atomic number belongs to the element {entity}?",
        "State the atomic number assigned to {entity}.",
        "In periodic-table notation, what atomic number identifies {entity}?",
        "Give the proton-count atomic number for {entity}.",
    ),
    "national_capital": (
        "What is the national capital of {entity}?",
        "Identify {entity}'s capital city.",
        "Which city is the capital of {entity}?",
        "Name the seat-of-government capital of {entity}.",
        "For {entity}, give the national capital.",
        "State the capital city associated with {entity}.",
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
    chemistry = "atomic number" in lowered or "proton-count" in lowered
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
