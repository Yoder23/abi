"""Registered R17 English frames, prompts, and independent functional oracle."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

MOODS = ("declarative", "question")
TENSES = ("present", "past", "future")
POLARITIES = ("positive", "negative")
NUMBERS = ("singular", "plural")
SIGNATURES = tuple("|".join(values) for values in product(MOODS, TENSES, POLARITIES, NUMBERS))
SLOT_KEYS = ("subject", "verb_base", "verb_3sg", "verb_past", "object")


@dataclass(frozen=True)
class Lexeme:
    singular_subject: str
    plural_subject: str
    verb_base: str
    verb_3sg: str
    verb_past: str
    object: str


EXTRACTION_LEXEMES = (
    Lexeme("Ava", "Ava and Bo", "admire", "admires", "admired", "the amber lantern"),
    Lexeme("Mira", "Mira and Niko", "inspect", "inspects", "inspected", "the quiet alcove"),
    Lexeme("Tari", "Tari and Uma", "carry", "carries", "carried", "the violet parcel"),
    Lexeme("Zara", "Zara and Ivo", "repair", "repairs", "repaired", "the silver compass"),
    Lexeme("Lena", "Lena and Oren", "observe", "observes", "observed", "the cedar doorway"),
    Lexeme("Kian", "Kian and Esme", "follow", "follows", "followed", "the narrow pathway"),
)

EVALUATION_LEXEMES = (
    Lexeme("Rhea", "Rhea and Sol", "measure", "measures", "measured", "the copper ribbon"),
    Lexeme("Pia", "Pia and Leon", "visit", "visits", "visited", "the mossy courtyard"),
    Lexeme("Noa", "Noa and Eira", "collect", "collects", "collected", "the ivory token"),
    Lexeme("Sana", "Sana and Remy", "paint", "paints", "painted", "the blue archway"),
)


class R17FrameError(RuntimeError):
    """Raised when a registered R17 semantic frame is malformed."""


def signature(mood: str, tense: str, polarity: str, number: str) -> str:
    value = "|".join((mood, tense, polarity, number))
    if value not in SIGNATURES:
        raise R17FrameError("unknown R17 feature signature")
    return value


def build_frame(sig: str, lexeme: Lexeme) -> dict[str, Any]:
    if sig not in SIGNATURES:
        raise R17FrameError("unknown R17 feature signature")
    mood, tense, polarity, number = sig.split("|")
    return {
        "signature": sig,
        "features": {
            "mood": mood,
            "tense": tense,
            "polarity": polarity,
            "subject_number": number,
        },
        "slots": {
            "subject": lexeme.singular_subject if number == "singular" else lexeme.plural_subject,
            "verb_base": lexeme.verb_base,
            "verb_3sg": lexeme.verb_3sg,
            "verb_past": lexeme.verb_past,
            "object": lexeme.object,
        },
    }


def prompt(frame: dict[str, Any]) -> str:
    features = frame["features"]
    slots = frame["slots"]
    return (
        "Realize the supplied semantic frame as exactly one natural English sentence. "
        "Use only the supplied lexical content, add no facts, use no contractions, and "
        "return only the sentence.\n"
        f"MOOD: {features['mood']}\n"
        f"TENSE: {features['tense']}\n"
        f"POLARITY: {features['polarity']}\n"
        f"SUBJECT_NUMBER: {features['subject_number']}\n"
        f"SUBJECT: {slots['subject']}\n"
        f"VERB_BASE: {slots['verb_base']}\n"
        f"VERB_THIRD_PERSON_SINGULAR: {slots['verb_3sg']}\n"
        f"VERB_PAST: {slots['verb_past']}\n"
        f"OBJECT: {slots['object']}"
    )


def expected(frame: dict[str, Any]) -> str:
    mood, tense, polarity, number = str(frame["signature"]).split("|")
    slots = frame["slots"]
    subject = str(slots["subject"])
    base = str(slots["verb_base"])
    third = str(slots["verb_3sg"])
    past = str(slots["verb_past"])
    obj = str(slots["object"])
    if mood == "declarative":
        if tense == "present" and polarity == "positive":
            verb = third if number == "singular" else base
            return f"{subject} {verb} {obj}."
        if tense == "present":
            auxiliary = "does" if number == "singular" else "do"
            return f"{subject} {auxiliary} not {base} {obj}."
        if tense == "past" and polarity == "positive":
            return f"{subject} {past} {obj}."
        if tense == "past":
            return f"{subject} did not {base} {obj}."
        if polarity == "positive":
            return f"{subject} will {base} {obj}."
        return f"{subject} will not {base} {obj}."
    if tense == "present":
        auxiliary = "Does" if number == "singular" else "Do"
    elif tense == "past":
        auxiliary = "Did"
    else:
        auxiliary = "Will"
    negative = " not" if polarity == "negative" else ""
    return f"{auxiliary}{negative} {subject} {base} {obj}?"


def public_rows(split: str) -> list[dict[str, Any]]:
    if split not in {"extraction", "evaluation"}:
        raise R17FrameError("unknown R17 public split")
    lexemes = EXTRACTION_LEXEMES if split == "extraction" else EVALUATION_LEXEMES
    repeats = 3 if split == "extraction" else 2
    rows = []
    for signature_index, sig in enumerate(SIGNATURES):
        for repeat in range(repeats):
            lexeme = lexemes[(signature_index + repeat) % len(lexemes)]
            frame = build_frame(sig, lexeme)
            rows.append(
                {
                    "record_id": f"r17-public-{split}-{signature_index:02d}-{repeat}",
                    "split": split,
                    **frame,
                    "prompt": prompt(frame),
                    "expected": expected(frame),
                }
            )
    return rows
