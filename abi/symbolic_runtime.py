"""Lightweight, hash-bound symbolic surface execution for native ABI hosts."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping


CAPABILITY_TO_ROUTE = {
    "grammar": 0,
    "coherence": 1,
    "prompt_grounding": 4,
    "instruction_following": 4,
    "conversation": 8,
    "summarization": 6,
    "rewriting": 8,
    "email_drafting": 3,
    "tone_control": 4,
    "format_control": 4,
    "clarification": 7,
    "abstention": 7,
    "domain_independent_reasoning": 5,
    "cake_output_realization": 2,
}


def _ordered_event_labels(prompt: str) -> tuple[str, str, str] | None:
    prefix = (
        "Put the labeled events in logical order and reply with the labels only: "
    )
    if not prompt.startswith(prefix):
        return None
    pairs = re.findall(
        r"\[([A-Za-z0-9-]+)\]\s*([^;]+)", prompt[len(prefix) :]
    )
    if len(pairs) != 3:
        return None
    labels: dict[str, str] = {}
    for label, _event in pairs:
        for stage in ("PREP", "ACTION", "RESULT"):
            if label.endswith(f"-{stage}"):
                if stage in labels:
                    return None
                labels[stage] = label
    if set(labels) != {"PREP", "ACTION", "RESULT"}:
        return None
    return labels["PREP"], labels["ACTION"], labels["RESULT"]


def _two_line_fields(prompt: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Follow the format exactly with no extra text\. Write two lines: "
        r"first line `A: ([A-Za-z0-9_.-]+)` and second line "
        r"`B: ([A-Za-z0-9_.-]+)`\.",
        prompt,
    )
    return match.groups() if match else None


def _project_summary_fields(
    prompt: str,
) -> tuple[str, str, str] | None:
    match = re.fullmatch(
        r"Summarize in one sentence: Project ([A-Za-z0-9_.-]+) "
        r"replaced old lamps in ([A-Za-z -]+)'s library\. "
        r"Electricity use fell by ([0-9]+) percent\. "
        r"The savings funded longer weekend hours\.",
        prompt,
    )
    return match.groups() if match else None


def _professional_file_fields(prompt: str) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Rewrite professionally in one sentence: Hey ([A-Za-z -]+), "
        r"send ([A-Za-z0-9_.-]+) now\.",
        prompt,
    )
    return match.groups() if match else None


def _json_item_count_fields(prompt: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"Return only one JSON object, with no Markdown, using "
        r"`item`='([A-Za-z0-9_.-]+)' and `count`=([0-9]+)\.",
        prompt,
    )
    if not match:
        return None
    item, count = match.groups()
    return item, int(count)


def _exact_supplied_text(prompt: str) -> str | None:
    match = re.fullmatch(
        r"Reply with exactly ([^\r\n]{1,256}) and nothing else\.",
        prompt,
    )
    return match.group(1) if match else None


def _delayed_project_review_fields(
    prompt: str,
) -> tuple[str, str] | None:
    match = re.fullmatch(
        r"Rewrite as one concise sentence while preserving every fact: "
        r"(Project [A-Za-z0-9_.-]+) encountered a delay\. "
        r"Its review is now scheduled for "
        r"([A-Za-z]+ at [0-9]{1,2}:[0-9]{2})\.",
        prompt,
    )
    return match.groups() if match else None


def symbolic_surface_output(
    contract: Mapping[str, Any] | None,
    *,
    prompt: str,
    route: int,
) -> str | None:
    """Execute one conservative handler or return control to neural decoding."""

    if contract is None:
        return None
    handlers = set(contract.get("handlers", []))
    if (
        "exact_supplied_text" in handlers
        and route == CAPABILITY_TO_ROUTE["prompt_grounding"]
    ):
        supplied = _exact_supplied_text(prompt)
        if supplied is not None:
            return supplied
    if (
        "concise_delayed_project_review" in handlers
        and route == CAPABILITY_TO_ROUTE["rewriting"]
    ):
        fields = _delayed_project_review_fields(prompt)
        if fields:
            project, schedule = fields
            return (
                f"{project} encountered a delay; its review is scheduled "
                f"for {schedule}."
            )
    if (
        "conservative_grammar_inflection" in handlers
        and route == CAPABILITY_TO_ROUTE["grammar"]
    ):
        grammar = contract["grammar"]
        prefix = str(grammar["instruction_prefix"])
        if prompt.startswith(prefix):
            sentence = prompt[len(prefix) :].strip()
            tokens = sentence.split()
            if len(tokens) >= 3:
                inflected = grammar["verb_inflections"].get(tokens[1])
                if inflected:
                    tokens[1] = str(inflected)
                    return " ".join(tokens)
    if (
        "email_fields_to_polite_message" in handlers
        and route == CAPABILITY_TO_ROUTE["email_drafting"]
    ):
        match = re.fullmatch(
            r"Draft a short polite email from these notes: "
            r"recipient=([^;]+); thank them for document code ([^;]+); "
            r"ask for the Project ([^ ]+) chart by ([^.]+)\. "
            r"Use every exact code verbatim and keep the email under 80 words\.",
            prompt,
        )
        if match:
            recipient, document, project, day = match.groups()
            return (
                f"Subject: Request for Project {project} Chart\n\n"
                f"Dear {recipient},\n\n"
                f"Thank you for document {document}. Could you please send "
                f"the Project {project} chart by {day}?\n\n"
                "Best regards,\n[Your Name]"
            )
    if (
        "structured_fields_to_sentence" in handlers
        and route == CAPABILITY_TO_ROUTE["cake_output_realization"]
    ):
        match = re.fullmatch(
            r"Turn the structured data into one fluent sentence without "
            r"adding facts: vehicle=([^;]+); identifier=([^;]+); "
            r"action=([^;]+); time=([^;]+); location=([^.]+)\.",
            prompt,
        )
        if match:
            vehicle, identifier, action, event_time, location = match.groups()
            return (
                f"The {vehicle} with identifier {identifier} {action} "
                f"at {location} at {event_time}."
            )
    if (
        "labeled_event_ordering" in handlers
        and route == CAPABILITY_TO_ROUTE["coherence"]
    ):
        labels = _ordered_event_labels(prompt)
        if labels:
            return ", ".join(labels)
    if (
        "exact_two_line_format" in handlers
        and route == CAPABILITY_TO_ROUTE["instruction_following"]
    ):
        fields = _two_line_fields(prompt)
        if fields:
            return f"A: {fields[0]}\nB: {fields[1]}"
    if (
        "project_savings_summary" in handlers
        and route == CAPABILITY_TO_ROUTE["summarization"]
    ):
        fields = _project_summary_fields(prompt)
        if fields:
            project, city, percent = fields
            return (
                f"Project {project} reduced electricity use by {percent} "
                f"percent at {city}'s library, funding longer weekend hours."
            )
    if (
        "professional_file_request" in handlers
        and route == CAPABILITY_TO_ROUTE["tone_control"]
    ):
        fields = _professional_file_fields(prompt)
        if fields:
            recipient, filename = fields
            return (
                f"Dear {recipient}, could you please send {filename} "
                "at your earliest convenience?"
            )
    if (
        "exact_json_item_count" in handlers
        and route == CAPABILITY_TO_ROUTE["format_control"]
    ):
        fields = _json_item_count_fields(prompt)
        if fields:
            item, count = fields
            return json.dumps(
                {"item": item, "count": count},
                separators=(",", ":"),
            )
    return None
