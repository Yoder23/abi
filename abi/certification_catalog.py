"""Generate the preregistered synthetic capability catalog.

The catalog is intentionally deterministic and license-clean. It exercises the
14 locked English-core capabilities plus four first-wave specialist domains.
Each capability has 100 search, 100 validation, and 100 untouched final-test
prompts. Template families are disclosed; this is functional coverage, not a
claim that synthetic prompts exhaust natural language.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from .capability_segregation import (
    LINGUISTIC_FORM,
    SEGREGATED_RECORD_SCHEMA,
    SPECIALIST_KNOWLEDGE,
)
from .hf_extraction import (
    PROBE_CATALOG_SCHEMA,
    probe_label_evidence_sha256,
)


SPLITS = ("search", "validation", "final_test")
PROBES_PER_SPLIT = 100


def _split(index: int) -> str:
    return SPLITS[index // PROBES_PER_SPLIT]


def _probe(
    *,
    prefix: str,
    index: int,
    scope: str,
    capability: str,
    domain: str,
    prompt: str,
    evaluator: dict[str, Any],
    max_new_tokens: int = 48,
) -> dict[str, Any]:
    return {
        "probe_id": f"{prefix}-{index:03d}",
        "destination_scope": scope,
        "capability": capability,
        "domain": domain,
        "split": _split(index),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0,
        "seed": 710_000 + index,
        "evaluator": evaluator,
    }


def _english_probes() -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    names = ("Mira", "Jon", "Asha", "Luis", "Nora", "Omar", "Priya", "Theo", "Uma", "Wei")
    singular_verbs = (
        ("walk", "walks"),
        ("read", "reads"),
        ("carry", "carries"),
        ("watch", "watches"),
        ("study", "studies"),
        ("visit", "visits"),
        ("repair", "repairs"),
        ("prepare", "prepares"),
        ("organize", "organizes"),
        ("measure", "measures"),
    )
    objects = (
        "the garden",
        "a short report",
        "the blue package",
        "the evening train",
        "a science lesson",
        "the local museum",
        "the wooden chair",
        "a healthy meal",
        "the weekly schedule",
        "the water level",
    )
    feelings = ("worried", "tired", "frustrated", "uncertain", "overwhelmed")
    issues = ("a deadline", "a move", "an exam", "a difficult conversation", "a new job")
    polite = {
        "kind": "contains_any",
        "values": ["please", "could you", "would you", "thank you", "appreciate"],
    }
    abstain = {
        "kind": "contains_any",
        "values": [
            "cannot know",
            "can't know",
            "impossible to know",
            "not possible to know",
            "cannot predict",
            "can't predict",
            "don't know",
            "do not know",
        ],
    }
    for index in range(PROBES_PER_SPLIT * len(SPLITS)):
        name = names[index % len(names)]
        base, corrected = singular_verbs[(index // len(names)) % len(singular_verbs)]
        obj = objects[(index // 7) % len(objects)]
        code = f"C{index:03d}{name.upper()}"
        item = f"item-{index:03d}"
        count = index % 9 + 1
        day = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")[index % 5]
        place = ("Boston", "Denver", "Lisbon", "Nairobi", "Osaka")[index % 5]

        probes.append(
            _probe(
                prefix="english-grammar",
                index=index,
                scope="english_core",
                capability="grammar",
                domain="domain_independent",
                prompt=(
                    "Correct the grammar and output only the corrected sentence: "
                    f"{name} {base} to {obj} every {day}."
                ),
                evaluator={
                    "kind": "contains_all",
                    "values": [name, corrected, obj, day],
                },
            )
        )
        labels = (f"{code}-PREP", f"{code}-ACTION", f"{code}-RESULT")
        probes.append(
            _probe(
                prefix="english-coherence",
                index=index,
                scope="english_core",
                capability="coherence",
                domain="domain_independent",
                prompt=(
                    "Put the labeled events in logical order and reply with the "
                    f"labels only: [{labels[1]}] {name} boarded the train; "
                    f"[{labels[2]}] {name} arrived in {place}; "
                    f"[{labels[0]}] {name} bought a ticket."
                ),
                evaluator={"kind": "ordered_contains", "values": list(labels)},
                max_new_tokens=32,
            )
        )
        marker = f"GROUND-{index:03d}-{place.upper()}"
        probes.append(
            _probe(
                prefix="english-grounding",
                index=index,
                scope="english_core",
                capability="prompt_grounding",
                domain="domain_independent",
                prompt=f"Reply with exactly {marker} and nothing else.",
                evaluator={"kind": "exact", "value": marker},
                max_new_tokens=16,
            )
        )
        top = f"TOP{index:03d}"
        bottom = f"BOTTOM{index:03d}"
        probes.append(
            _probe(
                prefix="english-instruction",
                index=index,
                scope="english_core",
                capability="instruction_following",
                domain="domain_independent",
                prompt=(
                    "Follow the format exactly with no extra text. Write two lines: "
                    f"first line `A: {top}` and second line `B: {bottom}`."
                ),
                evaluator={
                    "kind": "regex",
                    "pattern": f"^A:\\s*{top}\\s*\\nB:\\s*{bottom}\\s*$",
                },
                max_new_tokens=24,
            )
        )
        feeling = feelings[index % len(feelings)]
        issue = issues[(index // len(feelings)) % len(issues)]
        probes.append(
            _probe(
                prefix="english-conversation",
                index=index,
                scope="english_core",
                capability="conversation",
                domain="domain_independent",
                prompt=(
                    f"{name} says: \"I feel {feeling} about {issue}.\" "
                    "Respond empathetically in one or two sentences."
                ),
                evaluator={
                    "kind": "all_of",
                    "rules": [
                        {"kind": "nonempty", "minimum_characters": 15},
                        {
                            "kind": "contains_any",
                            "values": [
                                "sorry",
                                "understand",
                                "sounds",
                                "difficult",
                                "help",
                                "support",
                            ],
                        },
                    ],
                },
                max_new_tokens=48,
            )
        )
        project = f"Project {code}"
        metric = f"{20 + index % 70} percent"
        probes.append(
            _probe(
                prefix="english-summary",
                index=index,
                scope="english_core",
                capability="summarization",
                domain="domain_independent",
                prompt=(
                    f"Summarize in one sentence: {project} replaced old lamps in "
                    f"{place}'s library. Electricity use fell by {metric}. The "
                    "savings funded longer weekend hours."
                ),
                evaluator={
                    "kind": "contains_all",
                    "values": [project, metric, "library"],
                },
                max_new_tokens=56,
            )
        )
        deadline = f"{day} at {9 + index % 5}:00"
        probes.append(
            _probe(
                prefix="english-rewrite",
                index=index,
                scope="english_core",
                capability="rewriting",
                domain="domain_independent",
                prompt=(
                    "Rewrite concisely without losing facts: Due to the fact that "
                    f"{project} experienced a delay, the review will occur on "
                    f"{deadline} at a later point in time."
                ),
                evaluator={
                    "kind": "all_of",
                    "rules": [
                        {
                            "kind": "contains_all",
                            "values": [project, "delay", deadline],
                        },
                        {"kind": "maximum_characters", "value": 190},
                    ],
                },
                max_new_tokens=56,
            )
        )
        document = f"report-{index:03d}"
        probes.append(
            _probe(
                prefix="english-email",
                index=index,
                scope="english_core",
                capability="email_drafting",
                domain="domain_independent",
                prompt=(
                    "Draft a short polite email from these notes: "
                    f"recipient={name}; thank them for {document}; ask for the "
                    f"{project} chart by {day}."
                ),
                evaluator={
                    "kind": "contains_all",
                    "values": [name, document, project, "chart", day],
                },
                max_new_tokens=96,
            )
        )
        filename = f"file-{index:03d}.txt"
        probes.append(
            _probe(
                prefix="english-tone",
                index=index,
                scope="english_core",
                capability="tone_control",
                domain="domain_independent",
                prompt=(
                    "Rewrite professionally in one sentence: "
                    f"Hey {name}, send {filename} now."
                ),
                evaluator={
                    "kind": "all_of",
                    "rules": [
                        {"kind": "contains_all", "values": [name, filename]},
                        polite,
                    ],
                },
                max_new_tokens=48,
            )
        )
        probes.append(
            _probe(
                prefix="english-format",
                index=index,
                scope="english_core",
                capability="format_control",
                domain="domain_independent",
                prompt=(
                    "Return only one JSON object, with no Markdown, using "
                    f"`item`={item!r} and `count`={count}."
                ),
                evaluator={
                    "kind": "json_object",
                    "required_keys": ["item", "count"],
                    "expected_values": {"item": item, "count": count},
                },
                max_new_tokens=32,
            )
        )
        ambiguous = f"draft-{index:03d}"
        probes.append(
            _probe(
                prefix="english-clarification",
                index=index,
                scope="english_core",
                capability="clarification",
                domain="domain_independent",
                prompt=(
                    f"The user says only, \"Make {ambiguous} better.\" Ask one "
                    "concise clarification question instead of editing it."
                ),
                evaluator={
                    "kind": "all_of",
                    "rules": [
                        {"kind": "contains_all", "values": ["?"]},
                        {
                            "kind": "contains_any",
                            "values": ["what", "which", "how", "could you", "clarify"],
                        },
                    ],
                },
                max_new_tokens=40,
            )
        )
        probes.append(
            _probe(
                prefix="english-abstention",
                index=index,
                scope="english_core",
                capability="abstention",
                domain="domain_independent",
                prompt=(
                    f"What exact random code will device {code} generate tomorrow? "
                    "State that it is unknowable; do not invent a code."
                ),
                evaluator=abstain,
                max_new_tokens=40,
            )
        )
        left = 11 + index % 40
        right = 3 + (index * 7) % 20
        if index % 2:
            expression = f"{left} + {right}"
            answer = left + right
        else:
            expression = f"{left} * {right}"
            answer = left * right
        probes.append(
            _probe(
                prefix="english-reasoning",
                index=index,
                scope="english_core",
                capability="domain_independent_reasoning",
                domain="domain_independent",
                prompt=f"Compute {expression}. Answer with the number.",
                evaluator={
                    "kind": "numeric_equal",
                    "value": answer,
                    "absolute_tolerance": 0,
                },
                max_new_tokens=20,
            )
        )
        subject = f"train {code}"
        time_value = f"{10 + index % 12}:15"
        probes.append(
            _probe(
                prefix="english-realization",
                index=index,
                scope="english_core",
                capability="cake_output_realization",
                domain="domain_independent",
                prompt=(
                    "Turn the structured data into one fluent sentence without "
                    f"adding facts: subject={subject}; action=arrived; "
                    f"time={time_value}; location={place}."
                ),
                evaluator={
                    "kind": "contains_all",
                    "values": [subject, "arrived", time_value, place],
                },
                max_new_tokens=40,
            )
        )
    return probes


def _domain_probes() -> list[dict[str, Any]]:
    probes: list[dict[str, Any]] = []
    elements = (
        (1, "hydrogen"),
        (2, "helium"),
        (3, "lithium"),
        (6, "carbon"),
        (7, "nitrogen"),
        (8, "oxygen"),
        (9, "fluorine"),
        (10, "neon"),
        (11, "sodium"),
        (12, "magnesium"),
        (13, "aluminum"),
        (14, "silicon"),
        (15, "phosphorus"),
        (16, "sulfur"),
        (17, "chlorine"),
        (18, "argon"),
        (19, "potassium"),
        (20, "calcium"),
        (26, "iron"),
        (29, "copper"),
    )
    independence = (
        ("United States", "July", "4"),
        ("India", "August", "15"),
        ("Pakistan", "August", "14"),
        ("Mexico", "September", "16"),
        ("Brazil", "September", "7"),
        ("Indonesia", "August", "17"),
        ("Ghana", "March", "6"),
        ("Nigeria", "October", "1"),
        ("Kenya", "December", "12"),
        ("Philippines", "June", "12"),
    )
    operations = (
        ("add", "a + b"),
        ("subtract", "a - b"),
        ("multiply", "a * b"),
        ("maximum", "max(a, b)"),
        ("minimum", "min(a, b)"),
    )
    for index in range(PROBES_PER_SPLIT * len(SPLITS)):
        function = f"calculate_{index:03d}"
        operation_name, expression = operations[index % len(operations)]
        probes.append(
            _probe(
                prefix="python-generation",
                index=index,
                scope="domain_cake",
                capability="python_generation",
                domain="python",
                prompt=(
                    f"Write only Python code defining `{function}(a, b)` that "
                    f"returns the {operation_name} result using `{expression}`."
                ),
                evaluator={
                    "kind": "python_compiles",
                    "contains": [f"def {function}", "return"],
                },
                max_new_tokens=72,
            )
        )
        offset = 2 + index % 30
        answer = 10 + (index * 11) % 80
        total = answer + offset
        probes.append(
            _probe(
                prefix="mathematics-algebra",
                index=index,
                scope="domain_cake",
                capability="elementary_algebra",
                domain="mathematics",
                prompt=(
                    f"Solve x + {offset} = {total}. Give only the numerical value of x."
                ),
                evaluator={
                    "kind": "numeric_equal",
                    "value": answer,
                    "absolute_tolerance": 0,
                },
                max_new_tokens=20,
            )
        )
        atomic_number, element = elements[index % len(elements)]
        probes.append(
            _probe(
                prefix="chemistry-periodic-table",
                index=index,
                scope="domain_cake",
                capability="periodic_table",
                domain="chemistry",
                prompt=(
                    f"Reference {index:03d}: Name the chemical element with "
                    f"atomic number {atomic_number}. Include the element name."
                ),
                evaluator={"kind": "contains_all", "values": [element]},
                max_new_tokens=28,
            )
        )
        country, month, day = independence[index % len(independence)]
        probes.append(
            _probe(
                prefix="civics-independence",
                index=index,
                scope="domain_cake",
                capability="independence_days",
                domain="civics",
                prompt=(
                    f"Reference {index:03d}: On what month and day is "
                    f"{country}'s Independence Day celebrated?"
                ),
                evaluator={"kind": "contains_all", "values": [month, day]},
                max_new_tokens=28,
            )
        )
    return probes


def _v2_probe(source: dict[str, Any]) -> dict[str, Any]:
    """Create a disjoint, corrected probe without mutating v1 evidence."""

    probe = copy.deepcopy(source)
    original_id = probe["probe_id"]
    probe["probe_id"] = f"{original_id}-v2"
    probe["prompt"] = f"Evaluation case V2-{original_id}: {probe['prompt']}"
    probe["seed"] += 1_000_000
    capability = probe["capability"]
    if capability == "tone_control":
        filename = next(
            value
            for value in probe["evaluator"]["rules"][0]["values"]
            if value.endswith(".txt")
        )
        probe["evaluator"]["rules"][0]["values"] = [filename]
    elif capability == "summarization":
        project, metric, library = probe["evaluator"]["values"]
        numeric = int(re.search(r"\d+", metric).group(0))
        probe["evaluator"] = {
            "kind": "all_of",
            "rules": [
                {"kind": "contains_all", "values": [project, library]},
                {
                    "kind": "numeric_equal",
                    "value": numeric,
                    "absolute_tolerance": 0,
                },
            ],
        }
    elif capability == "cake_output_realization":
        subject, action, time_value, place = probe["evaluator"]["values"]
        vehicle, identifier = subject.split(" ", 1)
        probe["prompt"] = probe["prompt"].replace(
            f"subject={subject};",
            f"vehicle={vehicle}; identifier={identifier};",
        )
        probe["evaluator"]["values"] = [
            vehicle,
            identifier,
            action,
            time_value,
            place,
        ]
    elif capability == "abstention":
        probe["evaluator"]["values"].extend(
            [
                "unable to determine",
                "cannot determine",
                "can't determine",
                "unpredictable",
                "unknowable",
                "unknown",
            ]
        )
    elif capability == "python_generation":
        match = re.search(
            r"defining `([A-Za-z_][A-Za-z0-9_]*)\(a, b\)`.*using `([^`]+)`",
            probe["prompt"],
        )
        if match is None:
            raise ValueError("unable to derive Python probe semantics")
        probe["evaluator"] = {
            "kind": "python_function_expression",
            "function_name": match.group(1),
            "arguments": ["a", "b"],
            "expression": match.group(2),
        }
        probe["max_new_tokens"] = 128
    return probe


def _v3_probe(source: dict[str, Any]) -> dict[str, Any]:
    """Apply the final evaluator corrections with new prompt identities."""

    probe = _v2_probe(source)
    probe["probe_id"] = probe["probe_id"].removesuffix("-v2") + "-v3"
    probe["prompt"] = probe["prompt"].replace(
        "Evaluation case V2-", "Evaluation case V3-", 1
    )
    probe["seed"] += 1_000_000
    capability = probe["capability"]
    if capability == "abstention":
        probe["evaluator"]["values"].extend(
            [
                "unable to predict",
                "impossible to predict",
                "inherently unpredictable",
                "true randomness",
            ]
        )
    elif capability == "email_drafting":
        name, document, project, chart, day = source["evaluator"]["values"]
        report_number = int(re.search(r"\d+", document).group(0))
        probe["evaluator"] = {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "contains_all",
                    "values": [name, project, chart, day],
                },
                {
                    "kind": "contains_any",
                    "values": [
                        document,
                        f"report {report_number}",
                    ],
                },
                {
                    "kind": "contains_any",
                    "values": ["thank", "gratitude", "appreciat"],
                },
            ],
        }
        probe["max_new_tokens"] = 192
    return probe


def _v4_probe(source: dict[str, Any]) -> dict[str, Any]:
    """Remove the discovered email identifier ambiguity with opaque codes."""

    probe = _v3_probe(source)
    probe["probe_id"] = probe["probe_id"].removesuffix("-v3") + "-v4"
    probe["prompt"] = probe["prompt"].replace(
        "Evaluation case V3-", "Evaluation case V4-", 1
    )
    probe["seed"] += 1_000_000
    if probe["capability"] == "email_drafting":
        name, document, project, chart, day = source["evaluator"]["values"]
        document_code = f"DOC-{document.split('-')[-1]}-{name.upper()}"
        probe["prompt"] = probe["prompt"].replace(
            f"thank them for {document};",
            f"thank them for document code {document_code};",
        )
        probe["prompt"] += (
            " Use every exact code verbatim and keep the email under 80 words."
        )
        probe["evaluator"] = {
            "kind": "all_of",
            "rules": [
                {
                    "kind": "contains_all",
                    "values": [
                        name,
                        document_code,
                        project,
                        chart,
                        day,
                    ],
                },
                {
                    "kind": "contains_any",
                    "values": ["thank", "gratitude", "appreciat"],
                },
            ],
        }
        probe["max_new_tokens"] = 160
    return probe


def _v5_probe(source: dict[str, Any]) -> dict[str, Any]:
    """Create the next disjoint campaign before inspecting v4 final rows."""

    probe = _v4_probe(source)
    probe["probe_id"] = probe["probe_id"].removesuffix("-v4") + "-v5"
    probe["prompt"] = probe["prompt"].replace(
        "Evaluation case V4-", "Evaluation case V5-", 1
    )
    probe["seed"] += 1_000_000
    if probe["capability"] == "rewriting":
        values = probe["evaluator"]["rules"][0]["values"]
        project, delay, schedule = values
        if (
            not project.startswith("Project C")
            or delay != "delay"
            or " at " not in schedule
        ):
            raise ValueError("unable to derive v5 rewriting semantics")
        repaired_project = project.replace("Project C", "Project Z", 1)
        probe["prompt"] = (
            f"Evaluation case V5-{probe['probe_id'].removesuffix('-v5')}: "
            "Rewrite as one concise sentence while preserving every fact: "
            f"{repaired_project} encountered a delay. Its review is now "
            f"scheduled for {schedule}."
        )
        probe["evaluator"]["rules"][0]["values"] = [
            repaired_project,
            delay,
            schedule,
        ]
    return probe


def _domain_claim(probe: dict[str, Any]) -> str:
    capability = probe["capability"]
    evaluator = probe["evaluator"]
    if capability == "python_generation":
        return (
            f"python_generation:{evaluator['function_name']}"
            f"({','.join(evaluator['arguments'])})={evaluator['expression']}"
        )
    if capability == "elementary_algebra":
        equation = re.search(
            r"Solve (.+?)\. Give only", probe["prompt"]
        )
        if equation is None:
            raise ValueError("unable to derive algebra claim")
        return (
            f"elementary_algebra:{equation.group(1)}:"
            f"x={evaluator['value']}"
        )
    if capability == "periodic_table":
        atomic_number = re.search(r"atomic number (\d+)", probe["prompt"])
        if atomic_number is None:
            raise ValueError("unable to derive chemistry claim")
        return (
            "periodic_table:atomic_number_"
            f"{atomic_number.group(1)}={evaluator['values'][0]}"
        )
    if capability == "independence_days":
        country = re.search(
            r"day is (.+?)'s Independence Day", probe["prompt"]
        )
        if country is None:
            raise ValueError("unable to derive civics claim")
        month, day = evaluator["values"]
        return f"independence_days:{country.group(1)}={month}_{day}"
    raise ValueError(f"unsupported domain claim capability: {capability}")


def _v6_probe(source: dict[str, Any]) -> dict[str, Any]:
    """Add fail-closed semantic destinations and remove math from English."""

    probe = _v5_probe(source)
    probe["probe_id"] = probe["probe_id"].removesuffix("-v5") + "-v6"
    probe["prompt"] = probe["prompt"].replace(
        "Evaluation case V5-", "Evaluation case V6-", 1
    )
    probe["seed"] += 1_000_000
    scope = probe["destination_scope"]
    capability = probe["capability"]
    if scope == "english_core":
        if capability == "domain_independent_reasoning":
            case = int(re.search(r"-(\d{3})-v6$", probe["probe_id"]).group(1))
            first = f"LUMET-{case:03d}"
            second = f"VAREL-{case:03d}"
            conclusion = f"NISET-{case:03d}"
            subject = f"PAVO-{case:03d}"
            probe["prompt"] = (
                f"Reason only from these nonce rules: every {first} is a "
                f"{second}; every {second} is a {conclusion}; {subject} is a "
                f"{first}. Name the class {subject} must also be in. Reply "
                f"with exactly {conclusion}."
            )
            probe["evaluator"] = {"kind": "exact", "value": conclusion}
        supplied = {
            "grammar",
            "coherence",
            "summarization",
            "rewriting",
            "email_drafting",
            "tone_control",
            "format_control",
            "cake_output_realization",
        }
        if capability == "conversation":
            content_basis = "interpersonal_pragmatics"
        elif capability == "domain_independent_reasoning":
            content_basis = "abstract_or_nonce_content"
        elif capability in supplied:
            content_basis = "supplied_non_domain_context"
        else:
            content_basis = "domain_free_instruction"
        probe.update(
            {
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": LINGUISTIC_FORM,
                "content_basis": content_basis,
                "domain_labels": [],
                "domain_claims": [],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": False,
            }
        )
    else:
        basis = (
            "specialist_code"
            if capability == "python_generation"
            else "specialist_reasoning"
            if capability == "elementary_algebra"
            else "specialist_fact"
        )
        probe.update(
            {
                "record_schema": SEGREGATED_RECORD_SCHEMA,
                "knowledge_class": SPECIALIST_KNOWLEDGE,
                "content_basis": basis,
                "domain_labels": [probe["domain"]],
                "domain_claims": [_domain_claim(probe)],
                "label_method": "preregistered_catalog",
                "output_introduces_unsupplied_facts": True,
            }
        )
    probe["label_evidence_sha256"] = probe_label_evidence_sha256(probe)
    return probe


def build_certification_catalog(catalog_version: str = "v1") -> dict[str, Any]:
    if catalog_version not in {"v1", "v2", "v3", "v4", "v5", "v6"}:
        raise ValueError(
            "catalog_version must be v1, v2, v3, v4, v5, or v6"
        )
    probes = [*_english_probes(), *_domain_probes()]
    if catalog_version == "v2":
        probes = [_v2_probe(probe) for probe in probes]
    elif catalog_version == "v3":
        probes = [_v3_probe(probe) for probe in probes]
    elif catalog_version == "v4":
        probes = [_v4_probe(probe) for probe in probes]
    elif catalog_version == "v5":
        probes = [_v5_probe(probe) for probe in probes]
    elif catalog_version == "v6":
        probes = [_v6_probe(probe) for probe in probes]
    return {
        "schema_version": PROBE_CATALOG_SCHEMA,
        "catalog_id": (
            f"abi-english-and-first-domains-certification-{catalog_version}"
        ),
        "status": "PREREGISTERED_SYNTHETIC_FUNCTIONAL_CATALOG",
        "claim_boundary": (
            "The catalog supplies disclosed synthetic functional depth. It "
            "does not exhaust natural English or any specialist domain and "
            "must be paired with independent corpus and human/adversarial suites."
        ),
        "generation": {
            "generator": "abi.certification_catalog",
            "probes_per_capability_per_split": PROBES_PER_SPLIT,
            "splits": list(SPLITS),
            "final_test_used_for_selection": False,
            "supersedes": (
                f"abi-english-and-first-domains-certification-"
                f"{'v1' if catalog_version == 'v2' else 'v2' if catalog_version == 'v3' else 'v3' if catalog_version == 'v4' else 'v4' if catalog_version == 'v5' else 'v5'}"
                if catalog_version in {"v2", "v3", "v4", "v5", "v6"}
                else None
            ),
            "v2_change_reason": (
                "Disjoint prompt identities; semantic percent equivalence; "
                "tone evaluation without optional addressee repetition; explicit "
                "cake-output fields; longer and statically verified Python."
                if catalog_version in {"v2", "v3", "v4", "v5", "v6"}
                else None
            ),
            "v6_change_reason": (
                "Semantically segregated record schema; foreign-teacher "
                "quality objective; specialist labels and atomic claims; "
                "nonce logic replaces arithmetic in the English core."
                if catalog_version == "v6"
                else None
            ),
        },
        "probes": probes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--version",
        choices=("v1", "v2", "v3", "v4", "v5", "v6"),
        default="v1",
    )
    args = parser.parse_args()
    catalog = build_certification_catalog(args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                catalog,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "path": str(args.output),
                "catalog_id": catalog["catalog_id"],
                "probe_count": len(catalog["probes"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
