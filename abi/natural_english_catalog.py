"""Build the disjoint natural-paraphrase English acquisition catalog.

Unlike the historical certification catalogs, this campaign varies both the
surface instruction and the supplied content within every capability.  It
contains no closed-book specialist questions.  Every answer is either form
only, grounded in supplied text, interpersonal language, or nonce reasoning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from .capability_segregation import LINGUISTIC_FORM, SEGREGATED_RECORD_SCHEMA
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    load_probe_catalog,
    probe_label_evidence_sha256,
)


SPLITS = ("search", "validation", "final_test")
PROBES_PER_CAPABILITY_SPLIT = 100

NAMES = (
    "Mira",
    "Jon",
    "Asha",
    "Luis",
    "Nora",
    "Omar",
    "Priya",
    "Theo",
    "Uma",
    "Wei",
)
PLACES = (
    "the quiet room",
    "the east hall",
    "the garden",
    "the reading area",
    "the front desk",
)
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
TIMES = ("08:15", "09:30", "11:45", "14:20", "16:05")
FEELINGS = ("worried", "tired", "frustrated", "uncertain", "overwhelmed")
OBJECTS = ("blue folder", "small parcel", "meeting note", "green notebook", "draft")

WRAPPERS: dict[str, tuple[Callable[[str], str], ...]] = {
    "search": (
        lambda body: body,
        lambda body: f"Please complete this request: {body}",
        lambda body: f"Task for your next response: {body}",
        lambda body: f"Follow this instruction carefully: {body}",
    ),
    "validation": (
        lambda body: f"Can you help with the following? {body}",
        lambda body: f"Respond to this request: {body}",
        lambda body: f"I need you to do this: {body}",
        lambda body: f"Here is the instruction—{body}",
    ),
    "final_test": (
        lambda body: f"Would you handle this request? {body}",
        lambda body: f"Your next response should satisfy this: {body}",
        lambda body: f"Do the following carefully: {body}",
        lambda body: f"A user asks for this: {body}",
    ),
}


def _all_of(*rules: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "all_of", "rules": list(rules)}


def _contains_all(*values: str) -> dict[str, Any]:
    return {"kind": "contains_all", "values": list(values)}


def _contains_any(*values: str) -> dict[str, Any]:
    return {"kind": "contains_any", "values": list(values)}


def _case(index: int) -> dict[str, Any]:
    return {
        "name": NAMES[index % len(NAMES)],
        "other": NAMES[(index + 3) % len(NAMES)],
        "place": PLACES[(index // 2) % len(PLACES)],
        "day": DAYS[(index // 3) % len(DAYS)],
        "time": TIMES[(index // 5) % len(TIMES)],
        "feeling": FEELINGS[(index // 7) % len(FEELINGS)],
        "object": OBJECTS[(index // 11) % len(OBJECTS)],
        "code": f"N{index:03d}{NAMES[index % len(NAMES)].upper()}",
        "number": index % 9 + 1,
    }


def _grammar(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    variants = (
        (
            f"Correct the agreement error and return one sentence only: "
            f"{value['name']} walk through {value['place']} every {value['day']}.",
            _contains_all(value["name"], "walks", value["place"], value["day"]),
        ),
        (
            f"Edit only what is needed to make this grammatical: "
            f"The two visitors is waiting in {value['place']}.",
            _contains_all("two visitors", "are", "waiting", value["place"]),
        ),
        (
            f"Rewrite the sentence with the verb in the past tense: "
            f"{value['name']} opens the {value['object']} at {value['time']}.",
            _contains_all(value["name"], "opened", value["object"], value["time"]),
        ),
        (
            f"Fix the sentence and give no explanation: "
            f"{value['name']} and {value['other']} was ready on {value['day']}.",
            _contains_all(value["name"], value["other"], "were", value["day"]),
        ),
    )
    body, evaluator = variants[family]
    return body, evaluator, 48


def _coherence(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    code = value["code"]
    sequences = (
        (
            ("PREP", "ACT", "DONE"),
            (
                f"[{code}-ACT] {value['name']} entered the room; "
                f"[{code}-DONE] the conversation ended; "
                f"[{code}-PREP] {value['name']} opened the door."
            ),
        ),
        (
            ("START", "MIDDLE", "END"),
            (
                f"[{code}-END] the parcel was sealed; "
                f"[{code}-START] the box was opened; "
                f"[{code}-MIDDLE] the note was placed inside."
            ),
        ),
        (
            ("FIRST", "NEXT", "LAST"),
            (
                f"[{code}-NEXT] {value['name']} read the message; "
                f"[{code}-LAST] {value['name']} replied; "
                f"[{code}-FIRST] the message arrived."
            ),
        ),
        (
            ("ONE", "TWO", "THREE"),
            (
                f"[{code}-THREE] the lights went out; "
                f"[{code}-ONE] the last visitor left; "
                f"[{code}-TWO] the door was locked."
            ),
        ),
    )
    labels, events = sequences[family]
    ordered = [f"{code}-{label}" for label in labels]
    body = (
        "Put the event labels in logical order. Return the labels in order "
        f"without commentary: {events}"
    )
    return body, {"kind": "ordered_contains", "values": ordered}, 40


def _grounding(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    variants = (
        (
            f"Use only this note to answer. Note: “{value['name']} placed the "
            f"{value['object']} in {value['place']}.” Who placed what, and where?",
            _contains_all(value["name"], value["object"], value["place"]),
        ),
        (
            f"Read the supplied memo and answer from it alone: “Review "
            f"{value['code']} moved to {value['day']} at {value['time']}.” "
            "What code, day, and time are named?",
            _contains_all(value["code"], value["day"], value["time"]),
        ),
        (
            f"Context: “The label on the {value['object']} is {value['code']}.” "
            "State the object and its label using only that context.",
            _contains_all(value["object"], value["code"]),
        ),
        (
            f"According to this message only—“{value['other']} will meet "
            f"{value['name']} in {value['place']}.”—who will meet whom and where?",
            _contains_all(value["other"], value["name"], value["place"]),
        ),
    )
    body, evaluator = variants[family]
    return body, evaluator, 64


def _instruction(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    exact = (
        f"{value['code']} ready",
        f"left={value['name']}\nright={value['other']}",
        f"<{value['object'].replace(' ', '-')}>",
        f"{value['day'].upper()}|{value['time']}",
    )[family]
    bodies = (
        f"Reply with exactly these words and no punctuation: {exact}",
        f"Return exactly two lines, with no extra text:\n{exact}",
        f"Output only this bracketed text: {exact}",
        f"Copy this exact uppercase-and-time value with nothing else: {exact}",
    )
    return bodies[family], {"kind": "exact", "value": exact}, 32


def _conversation(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    variants = (
        (
            f"{value['name']} says, “I feel {value['feeling']} about tomorrow.” "
            "Reply warmly in one or two sentences.",
            _contains_any("understand", "sounds", "sorry", "help", "support"),
        ),
        (
            f"A new participant says, “Hi, I’m {value['name']}.” Offer a friendly "
            "one-sentence welcome.",
            _contains_any("welcome", "hello", "glad", "help"),
        ),
        (
            f"{value['name']} says, “Thank you for listening.” Respond naturally "
            "and briefly.",
            _contains_any("welcome", "glad", "anytime", "course", "happy"),
        ),
        (
            f"{value['name']} says, “I made a mistake and feel bad.” Give a kind, "
            "nonjudgmental response in no more than two sentences.",
            _contains_any("happens", "understand", "sorry", "okay", "learn"),
        ),
    )
    body, semantic = variants[family]
    return body, _all_of({"kind": "nonempty", "minimum_characters": 15}, semantic), 64


def _summary(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    variants = (
        (
            f"{value['name']} moved {value['number']} chairs into {value['place']}. "
            f"{value['other']} arranged them before {value['time']}.",
            (value["name"], value["other"], str(value["number"]), value["place"]),
        ),
        (
            f"Draft {value['code']} was reviewed on {value['day']}. A shorter "
            f"version will be shared at {value['time']}.",
            (value["code"], value["day"], value["time"]),
        ),
        (
            f"The {value['object']} arrived in {value['place']}. {value['name']} "
            f"will collect it on {value['day']}.",
            (value["object"], value["place"], value["name"], value["day"]),
        ),
        (
            f"{value['name']} asked for a quiet meeting. {value['other']} reserved "
            f"{value['place']} for {value['time']}.",
            (value["name"], value["other"], value["place"], value["time"]),
        ),
    )
    text, required = variants[family]
    body = f"Summarize this supplied text in one clear sentence: {text}"
    return body, _contains_all(*required), 64


def _rewrite(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    text = (
        f"{value['name']} has the {value['object']}. It must reach "
        f"{value['other']} by {value['time']}.",
        f"There is a delay for {value['code']}. The new review day is {value['day']}.",
        f"The meeting is in {value['place']}. It begins at {value['time']}.",
        f"{value['other']} requested {value['number']} copies. {value['name']} "
        f"will bring them on {value['day']}.",
    )[family]
    required = (
        (value["name"], value["object"], value["other"], value["time"]),
        (value["code"], "delay", value["day"]),
        (value["place"], value["time"]),
        (value["other"], str(value["number"]), value["name"], value["day"]),
    )[family]
    body = (
        "Combine the supplied statements into one concise, fluent sentence "
        f"without dropping any detail: {text}"
    )
    return body, _all_of(_contains_all(*required), {"kind": "maximum_characters", "value": 220}), 72


def _email(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    actions = (
        f"thank {value['other']} for the {value['object']} and ask for {value['code']} by {value['day']}",
        f"tell {value['other']} that {value['code']} moved to {value['time']} and ask them to confirm",
        f"ask {value['other']} to bring the {value['object']} to {value['place']} on {value['day']}",
        f"thank {value['other']} for helping {value['name']} and propose a meeting at {value['time']}",
    )
    required = (
        (value["other"], value["object"], value["code"], value["day"]),
        (value["other"], value["code"], value["time"], "confirm"),
        (value["other"], value["object"], value["place"], value["day"]),
        (value["other"], value["name"], value["time"], "thank"),
    )[family]
    body = (
        f"Draft a short, polite email from {value['name']} with these notes: "
        f"{actions[family]}. Include a greeting and closing; add no new facts."
    )
    return body, _all_of(_contains_all(*required), _contains_any("hello", "hi", "dear"), _contains_any("regards", "best", "sincerely", "thank")), 128


def _tone(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    blunt = (
        f"{value['other']}, send the {value['object']} now.",
        f"You forgot {value['code']}. Fix it by {value['day']}.",
        f"Move our meeting to {value['time']}.",
        f"Bring {value['number']} copies to {value['place']}.",
    )[family]
    required = (
        (value["other"], value["object"]),
        (value["code"], value["day"]),
        (value["time"],),
        (str(value["number"]), value["place"]),
    )[family]
    body = f"Rewrite this as one professional, courteous sentence: “{blunt}”"
    return body, _all_of(_contains_all(*required), _contains_any("please", "could", "would", "appreciate", "thank")), 64


def _format(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    if family in (0, 2):
        expected = {"item": value["object"], "code": value["code"]}
        body = (
            "Return only a valid JSON object with exactly the supplied values: "
            f"item={value['object']!r}; code={value['code']!r}."
        )
    else:
        expected = {"name": value["name"], "count": value["number"]}
        body = (
            "Produce JSON only, without a code fence or explanation, using "
            f"name={value['name']!r} and count={value['number']}."
        )
    return body, {"kind": "json_object", "required_keys": list(expected), "expected_values": expected}, 48


def _clarification(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    vague = (
        f"Make the {value['object']} better.",
        f"Change the meeting for {value['name']}.",
        f"Send {value['code']} soon.",
        f"Prepare {value['number']} things for {value['other']}.",
    )[family]
    body = (
        f"The request is underspecified: “{vague}” Ask one concise clarification "
        "question instead of assuming missing details."
    )
    evaluator = _all_of(
        _contains_all("?"),
        _contains_any("what", "which", "when", "where", "how", "could you", "clarify"),
    )
    return body, evaluator, 56


def _abstention(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    questions = (
        f"What exact random code will device {value['code']} produce tomorrow?",
        f"Without seeing the sealed note, what sentence is written inside it?",
        f"Which choice will {value['name']} make next week if they have not decided?",
        f"What exact number is hidden behind the closed card labeled {value['code']}?",
    )
    body = (
        f"{questions[family]} Say that the answer cannot be known from the "
        "information given; do not invent an answer."
    )
    return body, _contains_any("cannot know", "can't know", "do not know", "don't know", "unknown", "cannot determine", "not enough information", "unknowable"), 64


def _reasoning(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    a = f"LUMA-{value['code']}"
    b = f"VERI-{value['code']}"
    c = f"NORU-{value['code']}"
    subject = f"PAVO-{value['code']}"
    variants = (
        f"Every {a} is a {b}. Every {b} is a {c}. {subject} is a {a}.",
        f"All {a} belong to {b}; all {b} belong to {c}; {subject} belongs to {a}.",
        f"If something is {a}, it is {b}. If it is {b}, it is {c}. {subject} is {a}.",
        f"The {a} group is inside {b}, and {b} is inside {c}. {subject} is in {a}.",
    )
    body = (
        f"Reason only from these nonce statements: {variants[family]} "
        f"Return exactly the final class {subject} must belong to."
    )
    return body, {"kind": "exact", "value": c}, 32


def _realization(index: int, family: int) -> tuple[str, dict[str, Any], int]:
    value = _case(index)
    fields = (
        f"actor={value['name']}; action=placed; object={value['object']}; location={value['place']}",
        f"item={value['code']}; state=ready; time={value['time']}; day={value['day']}",
        f"speaker={value['other']}; action=thanked; recipient={value['name']}; reason=help",
        f"object={value['object']}; action=arrived; location={value['place']}; count={value['number']}",
    )
    required = (
        (value["name"], "placed", value["object"], value["place"]),
        (value["code"], "ready", value["time"], value["day"]),
        (value["other"], "thanked", value["name"], "help"),
        (value["object"], "arrived", value["place"], str(value["number"])),
    )[family]
    body = (
        "Turn these supplied fields into one natural English sentence without "
        f"adding information: {fields[family]}"
    )
    return body, _contains_all(*required), 64


BUILDERS: dict[str, Callable[[int, int], tuple[str, dict[str, Any], int]]] = {
    "grammar": _grammar,
    "coherence": _coherence,
    "prompt_grounding": _grounding,
    "instruction_following": _instruction,
    "conversation": _conversation,
    "summarization": _summary,
    "rewriting": _rewrite,
    "email_drafting": _email,
    "tone_control": _tone,
    "format_control": _format,
    "clarification": _clarification,
    "abstention": _abstention,
    "domain_independent_reasoning": _reasoning,
    "cake_output_realization": _realization,
}

CONTENT_BASIS = {
    "conversation": "interpersonal_pragmatics",
    "domain_independent_reasoning": "abstract_or_nonce_content",
    "instruction_following": "domain_free_instruction",
    "clarification": "domain_free_instruction",
    "abstention": "domain_free_instruction",
}

COVERAGE_V4_FAMILIES = {
    "instruction_following": {0, 1},
    "rewriting": {1},
    "email_drafting": {0, 1, 2},
    "tone_control": {1, 3},
    "cake_output_realization": {3},
}

COVERAGE_V4_COACHING = {
    "instruction_following": (
        "This is an exact-output task: copy every requested character and "
        "line break, and emit nothing else."
    ),
    "rewriting": (
        "Your one-sentence rewrite must include the literal word delay, the "
        "exact supplied code, and the exact supplied review day."
    ),
    "email_drafting": (
        "Keep the complete email under 80 words and include every named "
        "person, object or code, date or time, action, greeting, and closing "
        "exactly as supplied."
    ),
    "tone_control": (
        "Preserve every supplied code, count, place, and day exactly, and use "
        "an explicitly courteous word such as please, could, would, "
        "appreciate, or thank."
    ),
    "cake_output_realization": (
        "Use the supplied numeric count as a digit and explicitly include the "
        "object, arrival action, and location in one sentence."
    ),
}


def _v2_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Repair source-eligibility defects measured on the v6 extraction."""

    updated = dict(probe)
    updated["probe_id"] = str(updated["probe_id"]).removesuffix("-v1") + "-v2"
    updated["seed"] = int(updated["seed"]) + 1_000_000
    capability = str(updated["capability"])
    if capability == "abstention":
        updated["evaluator"] = _contains_any(
            "cannot know",
            "can't know",
            "do not know",
            "don't know",
            "unknown",
            "cannot determine",
            "not enough information",
            "unknowable",
            "don't have the capability",
            "do not have the capability",
            "no way to know",
        )
    elif capability == "domain_independent_reasoning":
        conclusion = str(updated["evaluator"]["value"])
        updated["evaluator"] = _contains_all(conclusion)
        updated["max_new_tokens"] = 96
    elif capability == "format_control":
        expected = dict(updated["evaluator"]["expected_values"])
        keys = list(expected)
        first, second = keys
        first_value = expected[first]
        second_value = expected[second]
        updated["prompt"] = (
            "Return exactly two plain-text lines and no Markdown. The first "
            f"line must be `{first}: {first_value}` and the second line must "
            f"be `{second}: {second_value}`."
        )
        updated["evaluator"] = {
            "kind": "regex",
            "pattern": (
                rf"^\s*{first}:\s*{first_value}\s*\n"
                rf"{second}:\s*{second_value}\s*$"
            ),
        }
    updated["label_evidence_sha256"] = probe_label_evidence_sha256(updated)
    return updated


def _v3_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Repair the two remaining source-eligibility defects from natural v2."""

    updated = _v2_probe(probe)
    updated["probe_id"] = str(updated["probe_id"]).removesuffix("-v2") + "-v3"
    updated["seed"] = int(updated["seed"]) + 1_000_000
    capability = str(updated["capability"])
    if capability == "abstention":
        updated["evaluator"] = _contains_any(
            *updated["evaluator"]["values"],
            "cannot provide",
            "can't provide",
            "cannot predict",
            "can't predict",
            "not possible",
            "impossible",
            "cannot assist",
        )
    elif capability == "coherence":
        updated["max_new_tokens"] = 96
    updated["label_evidence_sha256"] = probe_label_evidence_sha256(updated)
    return updated


def build_catalog(catalog_version: str = "v1") -> dict[str, Any]:
    if catalog_version == "coverage-v4":
        base = build_catalog("v2-v3")
        probes = []
        for probe in base["probes"]:
            if probe["split"] != "search":
                continue
            capability = str(probe["capability"])
            if capability not in COVERAGE_V4_FAMILIES:
                continue
            local_index = int(
                str(probe["probe_id"]).rsplit("-", 2)[-2]
            )
            if local_index % 4 not in COVERAGE_V4_FAMILIES[capability]:
                continue
            updated = dict(probe)
            updated["probe_id"] = (
                str(updated["probe_id"]).rsplit("-", 1)[0]
                + "-coverage-v4"
            )
            updated["seed"] = int(updated["seed"]) + 4_000_000
            updated["prompt"] = (
                str(updated["prompt"])
                + " "
                + COVERAGE_V4_COACHING[capability]
            )
            updated["label_evidence_sha256"] = (
                probe_label_evidence_sha256(updated)
            )
            probes.append(updated)
        return {
            "schema_version": PROBE_CATALOG_SCHEMA,
            "catalog_id": "abi-natural-english-coverage-supplement-v4",
            "status": "PREREGISTERED_TRAINING_ONLY_COVERAGE_SUPPLEMENT",
            "claim_boundary": (
                "This search-only supplement fills source-supervision holes "
                "measured after the v2-v3 validation screen. It changes no "
                "evaluator, imports no validation or final-test response, and "
                "earns no promotion credit by itself."
            ),
            "generation": {
                "generator": "abi.natural_english_catalog",
                "base_catalog": (
                    "abi-natural-english-acquisition-v2-v3"
                ),
                "selected_capability_families": {
                    key: sorted(value)
                    for key, value in COVERAGE_V4_FAMILIES.items()
                },
                "search_only": True,
                "probe_count": len(probes),
                "new_evaluators": 0,
                "validation_or_final_probes": 0,
            },
            "probes": probes,
        }
    if catalog_version not in {"v1", "v2", "v3", "v2-v3"}:
        raise ValueError(
            "catalog_version must be v1, v2, v3, v2-v3, or coverage-v4"
        )
    probes: list[dict[str, Any]] = []
    for split_index, split in enumerate(SPLITS):
        offset = split_index * PROBES_PER_CAPABILITY_SPLIT
        for capability, builder in BUILDERS.items():
            for local_index in range(PROBES_PER_CAPABILITY_SPLIT):
                index = offset + local_index
                family = local_index % 4
                body, evaluator, maximum = builder(index, family)
                prompt = WRAPPERS[split][family](body)
                probe: dict[str, Any] = {
                    "probe_id": f"natural-{capability}-{split}-{local_index:03d}-v1",
                    "destination_scope": "english_core",
                    "capability": capability,
                    "domain": "domain_independent",
                    "split": split,
                    "prompt": prompt,
                    "max_new_tokens": maximum,
                    "temperature": 0,
                    "seed": 8_310_000 + index,
                    "evaluator": evaluator,
                    "record_schema": SEGREGATED_RECORD_SCHEMA,
                    "knowledge_class": LINGUISTIC_FORM,
                    "content_basis": CONTENT_BASIS.get(
                        capability, "supplied_non_domain_context"
                    ),
                    "domain_labels": [],
                    "domain_claims": [],
                    "label_method": "preregistered_catalog",
                    "output_introduces_unsupplied_facts": False,
                }
                probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
                probes.append(probe)
    if catalog_version == "v2":
        probes = [_v2_probe(probe) for probe in probes]
    elif catalog_version == "v3":
        probes = [_v3_probe(probe) for probe in probes]
    elif catalog_version == "v2-v3":
        probes = [
            (
                _v3_probe(probe)
                if probe["capability"] in {"abstention", "coherence"}
                else _v2_probe(probe)
            )
            for probe in probes
        ]
    catalog = {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": f"abi-natural-english-acquisition-{catalog_version}",
        "status": "PREREGISTERED_NATURAL_PARAPHRASE_ACQUISITION_CATALOG",
        "claim_boundary": (
            "This deterministic catalog tests diverse natural instruction forms "
            "over supplied, interpersonal, and nonce content. It is a bounded "
            "functional suite, not an exhaustive definition of English fluency."
        ),
        "generation": {
            "generator": "abi.natural_english_catalog",
            "capabilities": list(BUILDERS),
            "probes_per_capability_per_split": PROBES_PER_CAPABILITY_SPLIT,
            "natural_prompt_families_per_capability_per_split": 4,
            "template_identity_overlap_between_splits": 0,
            "final_test_used_for_selection": False,
            "specialist_or_closed_book_prompts": 0,
        },
        "probes": probes,
    }
    if catalog_version == "v2":
        catalog["generation"].update(
            {
                "supersedes": "abi-natural-english-acquisition-v1",
                "v2_change_reason": (
                "The completed v6 source survey measured semantically valid "
                "abstention wording outside the old evaluator, truncated nonce "
                "reasoning, and persistent JSON code fences. V2 broadens valid "
                "abstention language, gives reasoning enough output budget, and "
                "uses strict plain-text two-line format control. No candidate "
                "or natural final-test output informed these changes."
                ),
            }
        )
    elif catalog_version == "v3":
        catalog["generation"].update(
            {
                "supersedes": "abi-natural-english-acquisition-v2",
                "v3_change_reason": (
                    "The completed natural-v2 source survey measured valid "
                    "refusals phrased as cannot provide/predict or not possible, "
                    "and otherwise correct ordered-label responses truncated by "
                    "the 40-token ceiling. V3 expands only valid abstention "
                    "phrases and raises only coherence to 96 source tokens. No "
                    "candidate or natural final-test output informed the change."
                ),
            }
        )
    elif catalog_version == "v2-v3":
        catalog["generation"].update(
            {
                "composes": [
                    "abi-natural-english-acquisition-v2",
                    "abi-natural-english-acquisition-v3",
                ],
                "v2_capabilities": [
                    capability
                    for capability in BUILDERS
                    if capability not in {"abstention", "coherence"}
                ],
                "v3_capabilities": ["abstention", "coherence"],
                "composition_reason": (
                    "The preregistered natural-v3 protocol authorizes v3 only "
                    "for abstention and coherence and requires reuse of v2 for "
                    "the other twelve capabilities. This catalog is the exact "
                    "mixed-lineage validation and final-test surface matching "
                    "that training composition."
                ),
            }
        )
    return catalog


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3", "v2-v3", "coverage-v4"),
        default="v1",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error(f"catalog is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            build_catalog(args.version),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    load_probe_catalog(output)
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
