"""Prospectively frozen normalization and fresh development population."""

from __future__ import annotations

import random
import re
from pathlib import Path

from experiments.plan_english_r98_completion import core as previous

ROOT, PARENT, HOST = previous.ROOT, previous.PARENT, previous.HOST
PARENT_SHA, HOST_COMMIT = previous.PARENT_SHA, previous.HOST_COMMIT
SOURCE, REVISION = previous.SOURCE, previous.REVISION
TASKS, ARMS, SEED = previous.TASKS, previous.ARMS, 99001
STEPS, BATCH, CONTEXT, MAX_NEW = 384, 2, 1024, 640
HERE = Path(__file__).resolve().parent
OUT = ROOT / "results/grounded_english_r99"
sha, canonical, write_once = previous.sha, previous.canonical, previous.write_once
read, rows, append = previous.read, previous.rows, previous.append
model_prompt, target = previous.model_prompt, previous.target
visible_response, screen = previous.visible_response, previous.screen
aggregate, source_gate = previous.aggregate, previous.source_gate


def development_catalog():
    # Fresh identifiers, field assignments and order; same bounded task templates.
    # This is not claimed to be a new broad-English population.
    rng = random.Random(99002)
    result = []
    for original in previous.original.catalog("development"):
        row = {
            **original,
            "plan": {
                "facts": list(original["plan"]["facts"]),
                "request": original["plan"]["request"],
            },
            "required": list(original["required"]),
        }
        replacements = {row["code"]: row["code"].replace("NOTEV", "NOTEN99V")}
        for person in re.findall(r"\bPersonV[A-Za-z0-9]+\b", row["prompt"]):
            replacements[person] = person.replace("PersonV", "PersonN99V")
        old_day = next(
            day
            for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
            if day in row["prompt"]
        )
        replacements[old_day] = rng.choice(
            [
                day
                for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
                if day != old_day
            ]
        )

        def replace(text):
            for before, after in replacements.items():
                text = text.replace(before, after)
            return text

        row["id"] = original["id"].replace("r98-", "r99-")
        row["prompt"] = replace(row["prompt"])
        row["code"] = replace(row["code"])
        row["required"] = [replace(value) for value in row["required"]]
        row["plan"]["facts"] = [replace(value) for value in row["plan"]["facts"]]
        row["plan"]["request"] = replace(row["plan"]["request"])
        result.append(row)
    rng.shuffle(result)
    return result


def normalize(row: dict, teacher: str) -> dict:
    """Select source spans only. No evaluator, answer rewriting or model calls.

    The public user request supplies the required entity slots; these are not
    evaluation answers. Every emitted non-whitespace byte comes from source.
    Minimum cost means only word count among these source units for four literal
    constraints, never globally minimal English or semantic faithfulness.
    """
    if row["task"] != "email_drafting":
        return {"text": teacher, "spans": [[0, len(teacher)]], "mode": "identity"}
    units = []
    greeting, signoff = None, None
    for match in re.finditer(r"[^\n.!?]+(?:[.!?]|$)", teacher, re.M):
        start, end = match.span()
        while start < end and teacher[start].isspace():
            start += 1
        while end > start and teacher[end - 1].isspace():
            end -= 1
        text = teacher[start:end]
        if not text:
            continue
        if re.match(r"^(Dear|Hi|Hello)\b", text, re.I):
            greeting = greeting or (start, end)
            continue
        if re.fullmatch(
            r"(?:Best(?: regards)?|Kind regards|Regards|Sincerely|Thanks|Thank you)[,!:]?",
            text,
            re.I,
        ):
            signoff = (start, end)
            continue
        if text.startswith("Subject:"):
            continue
        mask = 0
        for i, literal in enumerate(row["required"]):
            if re.search(r"\b" + re.escape(literal) + r"\b", text, re.I):
                mask |= 1 << i
        if re.search(
            r"\b(pending|provisional|unconfirmed|awaiting|awaits)\b|not (?:yet )?(?:approved|confirmed)",
            text,
            re.I,
        ):
            mask |= 1 << len(row["required"])
        if mask:
            units.append((start, end, mask, len(re.findall(r"\b[\w']+\b", text))))
    # Each source sentence is available once; deterministic lexical tie-break.
    frontier = {0: (0, ())}
    for start, end, mask, cost in units:
        for covered, (old_cost, selected) in list(frontier.items()):
            proposal = (old_cost + cost, (*selected, (start, end)))
            new_mask = covered | mask
            if new_mask not in frontier or proposal < frontier[new_mask]:
                frontier[new_mask] = proposal
    complete = (1 << (len(row["required"]) + 1)) - 1
    if greeting is None or signoff is None or complete not in frontier:
        # Do not manufacture missing coverage or replace a failed teacher answer.
        return {
            "text": teacher,
            "spans": [[0, len(teacher)]],
            "mode": "unmodified_missing_coverage",
        }
    spans = sorted({greeting, signoff, *frontier[complete][1]})
    return {
        "text": "\n\n".join(teacher[start:end] for start, end in spans),
        "spans": [list(span) for span in spans],
        "mode": "source_sentence_selection",
    }


def verify_binding():
    manifest = read(OUT / "freeze.json")
    if not manifest.get("files"):
        raise RuntimeError("Missing R99 frozen inputs")
    for relative, digest in manifest["files"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"R99 frozen input changed or missing: {relative}")
    previous.verify_binding()
    return manifest
