"""Evidence-driven R17 source-interface repair; all scientific rows stay fixed."""

from __future__ import annotations

from typing import Any

from .frames import public_rows


def expected_v2(frame: dict[str, Any]) -> str:
    mood, tense, polarity, number = str(frame["signature"]).split("|")
    if mood != "question" or polarity != "negative":
        return str(frame["expected"])
    slots = frame["slots"]
    if tense == "present":
        auxiliary = "Does" if number == "singular" else "Do"
    elif tense == "past":
        auxiliary = "Did"
    else:
        auxiliary = "Will"
    return f"{auxiliary} {slots['subject']} not {slots['verb_base']} {slots['object']}?"


def prompt_v2(frame: dict[str, Any]) -> str:
    features = frame["features"]
    slots = frame["slots"]
    mood = str(features["mood"])
    tense = str(features["tense"])
    polarity = str(features["polarity"])
    instructions = [
        "Realize the supplied semantic frame as exactly one natural English sentence.",
        "Use only the supplied lexical content and add no facts.",
        "Return only the sentence and never explain your choice.",
        "Do not use contractions.",
        (
            "The output MUST be a yes/no question ending in a question mark."
            if mood == "question"
            else "The output MUST be a declarative sentence ending in a period."
        ),
        (
            "The output MUST explicitly include the separate word 'not'."
            if polarity == "negative"
            else "The output MUST be positive and must not include a negator."
        ),
        (
            "Use simple future tense with the auxiliary 'will'."
            if tense == "future"
            else f"Use simple {tense} tense."
        ),
    ]
    if mood == "question" and polarity == "negative":
        instructions.append("In the uncontracted question, place the subject before 'not'.")
    return (
        " ".join(instructions)
        + "\n"
        + f"SENTENCE_TYPE: {'yes/no question' if mood == 'question' else 'declarative'}\n"
        + f"TENSE: simple {tense}\n"
        + f"POLARITY: {polarity}\n"
        + f"SUBJECT_NUMBER: {features['subject_number']}\n"
        + f"SUBJECT: {slots['subject']}\n"
        + f"VERB_BASE: {slots['verb_base']}\n"
        + f"VERB_THIRD_PERSON_SINGULAR: {slots['verb_3sg']}\n"
        + f"VERB_PAST: {slots['verb_past']}\n"
        + f"OBJECT: {slots['object']}"
    )


def public_rows_v2(split: str) -> list[dict[str, Any]]:
    rows = []
    for original in public_rows(split):
        row = {**original}
        row["expected"] = expected_v2(row)
        row["prompt"] = prompt_v2(row)
        rows.append(row)
    return rows
