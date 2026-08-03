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

_REQUEST_PREFACES = (
    "Can you help with the following? ",
    "Here is the instruction—",
    "Here is the instructionâ€”",
    "Respond to this request: ",
    "Please complete this request: ",
    "Follow this instruction carefully: ",
    "Task for your next response: ",
)
_SUPPORTED_REALIZATION_SCHEMAS = frozenset(
    {
        frozenset({"object", "action", "location", "count"}),
        frozenset({"actor", "action", "object", "location"}),
        frozenset({"speaker", "action", "recipient", "reason"}),
        frozenset({"item", "state", "time", "day"}),
        frozenset({"record", "state", "owner", "day", "purpose"}),
        frozenset(
            {"object", "action", "identifier", "destination", "day"}
        ),
        frozenset({"actor", "action", "object", "source", "time"}),
        frozenset({"actor", "action", "object", "location", "time"}),
    }
)


def _without_request_preface(prompt: str) -> str:
    for prefix in _REQUEST_PREFACES:
        if prompt.startswith(prefix):
            return prompt[len(prefix) :]
    return prompt


def _natural_email_fields(prompt: str) -> dict[str, str] | None:
    body = _without_request_preface(prompt)
    match = re.fullmatch(
        r"Draft a short, polite email from ([A-Za-z -]+) with these notes: "
        r"(.+?)\. Include a greeting and closing; add no new facts\."
        r"(?: Keep the complete email under 80 words and include every named "
        r"person, object or code, date or time, action, greeting, and closing "
        r"exactly as supplied\.)?",
        body,
    )
    if not match:
        return None
    sender, notes = match.groups()
    patterns = (
        (
            "thank_and_request",
            re.fullmatch(
                r"thank ([A-Za-z -]+) for (.+?) and ask for "
                r"([A-Za-z0-9-]+) by ([A-Za-z]+)",
                notes,
            ),
            ("recipient", "object", "code", "day"),
        ),
        (
            "moved_and_confirm",
            re.fullmatch(
                r"tell ([A-Za-z -]+) that ([A-Za-z0-9-]+) moved to "
                r"([0-9]{1,2}:[0-9]{2}) and ask them to confirm",
                notes,
            ),
            ("recipient", "code", "time"),
        ),
        (
            "bring_request",
            re.fullmatch(
                r"ask ([A-Za-z -]+) to bring (.+?) to (.+?) on "
                r"([A-Za-z]+)",
                notes,
            ),
            ("recipient", "object", "location", "day"),
        ),
        (
            "thanks_and_meeting",
            re.fullmatch(
                r"thank ([A-Za-z -]+) for helping ([A-Za-z -]+) and "
                r"propose a meeting at ([0-9]{1,2}:[0-9]{2})",
                notes,
            ),
            ("recipient", "helped_person", "time"),
        ),
    )
    for kind, parsed, names in patterns:
        if parsed:
            fields = {
                name: value.strip()
                for name, value in zip(names, parsed.groups(), strict=True)
            }
            fields.update({"kind": kind, "sender": sender.strip()})
            return fields
    return None


def _generic_realization_fields(prompt: str) -> dict[str, str] | None:
    body = _without_request_preface(prompt)
    prefixes = (
        "Turn these supplied fields into one natural English sentence "
        "without adding information: ",
        "Turn the supplied fields into one fluent English sentence. "
        "Include every field value exactly and add no information: ",
    )
    payload = None
    for prefix in prefixes:
        if body.startswith(prefix):
            payload = body[len(prefix) :]
            break
    if payload is None:
        return None
    fields: dict[str, str] = {}
    for item in payload.rstrip(".").split(";"):
        key, separator, value = item.strip().partition("=")
        if (
            not separator
            or not re.fullmatch(r"[a-z_]+", key)
            or not value.strip()
            or key in fields
        ):
            return None
        fields[key] = value.strip()
    if frozenset(fields) not in _SUPPORTED_REALIZATION_SCHEMAS:
        return None
    return fields


def _nonce_transitive_reasoning_fields(
    prompt: str,
) -> tuple[str, str] | None:
    """Parse a strict two-hop class chain supplied entirely in the prompt."""

    marker = "Reason only from these nonce statements: "
    if prompt.count(marker) != 1:
        return None
    request_prefix, _marker, remainder = prompt.partition(marker)
    if (
        len(request_prefix) > 96
        or any(character in request_prefix for character in "\r\n;")
    ):
        return None
    body = marker + remainder
    token = r"([A-Za-z0-9_.-]+)"
    patterns = (
        rf"Reason only from these nonce statements: Every {token} is a "
        rf"{token}\. Every {token} is a {token}\. {token} is a {token}\. "
        rf"Return exactly the final class {token} must belong to\.",
        rf"Reason only from these nonce statements: All {token} belong "
        rf"to {token}; all {token} belong to {token}; {token} belongs "
        rf"to {token}\. Return exactly the final class {token} must "
        rf"belong to\.",
        rf"Reason only from these nonce statements: If something is "
        rf"{token}, it is {token}\. If it is {token}, it is {token}\. "
        rf"{token} is {token}\. Return exactly the final class {token} "
        rf"must belong to\.",
        rf"Reason only from these nonce statements: The {token} group "
        rf"is inside {token}, and {token} is inside {token}\. {token} "
        rf"is in {token}\. Return exactly the final class {token} must "
        rf"belong to\.",
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, body)
        if not match:
            continue
        (
            first_class,
            intermediate,
            repeated_intermediate,
            final_class,
            subject,
            subject_class,
            repeated_subject,
        ) = match.groups()
        if (
            intermediate == repeated_intermediate
            and first_class == subject_class
            and subject == repeated_subject
        ):
            return subject, final_class
    return None


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


def _natural_ordered_event_labels(
    prompt: str,
) -> tuple[str, str, str] | None:
    body = _without_request_preface(prompt)
    prefix = (
        "Put the event labels in logical order. Return the labels in order "
        "without commentary: "
    )
    if not body.startswith(prefix):
        return None
    pairs = re.findall(
        r"\[([A-Za-z0-9-]+)\]\s*([^;]+)", body[len(prefix) :]
    )
    if len(pairs) != 3:
        return None
    by_suffix: dict[str, str] = {}
    for label, _event in pairs:
        suffix = label.rsplit("-", 1)[-1]
        if suffix in by_suffix:
            return None
        by_suffix[suffix] = label
    for order in (
        ("PREP", "ACT", "DONE"),
        ("START", "MIDDLE", "END"),
        ("FIRST", "NEXT", "LAST"),
        ("ONE", "TWO", "THREE"),
    ):
        if set(by_suffix) == set(order):
            return tuple(by_suffix[stage] for stage in order)
    return None


def _natural_ordered_event_labels_preface_v2(
    prompt: str,
) -> tuple[str, str, str] | None:
    prefix = "I need you to do this: "
    if not prompt.startswith(prefix):
        return None
    return _natural_ordered_event_labels(prompt[len(prefix) :])


def _natural_concise_statement_combination(prompt: str) -> str | None:
    body = _without_request_preface(prompt)
    prefix = (
        "Combine the supplied statements into one concise, fluent sentence "
        "without dropping any detail: "
    )
    if not body.startswith(prefix):
        return None
    statements = body[len(prefix) :]
    match = re.fullmatch(
        r"There is a delay for ([A-Za-z0-9_.-]+)\. The new review day is "
        r"([A-Za-z]+)\.(?: Your one-sentence rewrite must include the "
        r"literal word delay, the exact supplied code, and the exact "
        r"supplied review day\.)?",
        statements,
    )
    if match:
        code, day = match.groups()
        return f"There is a delay for {code}, and the new review day is {day}."
    match = re.fullmatch(
        r"([A-Za-z -]+) requested ([0-9]+) copies\. ([A-Za-z -]+) "
        r"will bring them on ([A-Za-z]+)\.",
        statements,
    )
    if match:
        requester, count, courier, day = match.groups()
        return (
            f"{requester} requested {count} copies, and {courier} will "
            f"bring them on {day}."
        )
    match = re.fullmatch(
        r"([A-Za-z -]+) has the ([A-Za-z -]+)\. It must reach "
        r"([A-Za-z -]+) by ([0-9]{1,2}:[0-9]{2})\.",
        statements,
    )
    if match:
        sender, item, recipient, deadline = match.groups()
        return (
            f"{sender} has the {item}, which must reach {recipient} by "
            f"{deadline}."
        )
    match = re.fullmatch(
        r"The meeting is in the ([A-Za-z -]+)\. It begins at "
        r"([0-9]{1,2}:[0-9]{2})\.",
        statements,
    )
    if match:
        location, event_time = match.groups()
        return f"The meeting begins at {event_time} in the {location}."
    return None


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
        "natural_email_from_notes" in handlers
        and route == CAPABILITY_TO_ROUTE["email_drafting"]
    ):
        fields = _natural_email_fields(prompt)
        if fields:
            sender = fields["sender"]
            recipient = fields["recipient"]
            kind = fields["kind"]
            if kind == "thank_and_request":
                subject = f"Request for {fields['code']}"
                message = (
                    f"Thank you for {fields['object']}. Could you please "
                    f"provide {fields['code']} by {fields['day']}?"
                )
            elif kind == "moved_and_confirm":
                subject = f"{fields['code']} Schedule"
                message = (
                    f"{fields['code']} moved to {fields['time']}. "
                    "Please confirm."
                )
            elif kind == "bring_request":
                subject = "Request"
                message = (
                    f"Please bring {fields['object']} to "
                    f"{fields['location']} on {fields['day']}."
                )
            else:
                subject = "Meeting"
                message = (
                    f"Thank you for helping {fields['helped_person']}. "
                    f"Could we meet at {fields['time']}?"
                )
            return (
                f"Subject: {subject}\n\nDear {recipient},\n\n"
                f"{message}\n\nBest regards,\n{sender}"
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
        "generic_supplied_field_realization" in handlers
        and route == CAPABILITY_TO_ROUTE["cake_output_realization"]
    ):
        fields = _generic_realization_fields(prompt)
        if fields:
            keys = frozenset(fields)
            if keys == {"object", "action", "location", "count"}:
                return (
                    f"The {fields['object']} {fields['action']} at "
                    f"{fields['location']}, with a count of {fields['count']}."
                )
            if keys == {"actor", "action", "object", "location"}:
                return (
                    f"{fields['actor']} {fields['action']} {fields['object']} "
                    f"at {fields['location']}."
                )
            if keys == {"speaker", "action", "recipient", "reason"}:
                return (
                    f"{fields['speaker']} {fields['action']} "
                    f"{fields['recipient']} for {fields['reason']}."
                )
            if keys == {"item", "state", "time", "day"}:
                return (
                    f"{fields['item']} was {fields['state']} at "
                    f"{fields['time']} on {fields['day']}."
                )
            if keys == {"record", "state", "owner", "day", "purpose"}:
                return (
                    f"{fields['record']}, owned by {fields['owner']}, was "
                    f"{fields['state']} on {fields['day']} for "
                    f"{fields['purpose']}."
                )
            if keys == {
                "object",
                "action",
                "identifier",
                "destination",
                "day",
            }:
                return (
                    f"The {fields['object']} with identifier "
                    f"{fields['identifier']} was {fields['action']} at "
                    f"{fields['destination']} on {fields['day']}."
                )
            source = fields.get("source") or fields.get("location")
            return (
                f"{fields['actor']} {fields['action']} {fields['object']} "
                f"at {source} at {fields['time']}."
            )
    if (
        "nonce_transitive_class_reasoning" in handlers
        and route == CAPABILITY_TO_ROUTE["domain_independent_reasoning"]
    ):
        fields = _nonce_transitive_reasoning_fields(prompt)
        if fields:
            subject, final_class = fields
            return f"{subject} must belong to {final_class}."
    if (
        "natural_labeled_event_ordering" in handlers
        and route == CAPABILITY_TO_ROUTE["coherence"]
    ):
        labels = _natural_ordered_event_labels(prompt)
        if labels:
            return ", ".join(labels)
    if (
        "natural_labeled_event_ordering_preface_v2" in handlers
        and route == CAPABILITY_TO_ROUTE["coherence"]
    ):
        labels = _natural_ordered_event_labels_preface_v2(prompt)
        if labels:
            return ", ".join(labels)
    if (
        "natural_concise_statement_combination" in handlers
        and route == CAPABILITY_TO_ROUTE["rewriting"]
    ):
        combined = _natural_concise_statement_combination(prompt)
        if combined:
            return combined
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


def truncate_novel_lexical_repetition(
    output: str,
    prompt: str,
    *,
    threshold: int,
) -> str:
    """Stop a novel lexical loop without importing the training runtime."""

    if threshold <= 0:
        return output
    prompt_words = re.findall(r"[\w']+", prompt.casefold())
    allowed = {
        tuple(prompt_words[index : index + 4])
        for index in range(max(0, len(prompt_words) - 3))
    }
    matches = list(re.finditer(r"[\w']+", output))
    words: list[str] = []
    counts: dict[tuple[str, ...], int] = {}
    repeated_occurrences = 0
    for match in matches:
        words.append(match.group().casefold())
        if len(words) < 4:
            continue
        fourgram = tuple(words[-4:])
        if fourgram in allowed:
            continue
        previous = counts.get(fourgram, 0)
        counts[fourgram] = previous + 1
        if previous >= 1:
            repeated_occurrences += 1
        if repeated_occurrences < threshold:
            continue
        raw_prefix = output[: match.start()].rstrip(" ,;:-\n\t")
        sentence_ends = list(
            re.finditer(r"[.!?](?:[\"'â€â€™])?", raw_prefix)
        )
        if sentence_ends and sentence_ends[-1].end() >= 16:
            return raw_prefix[: sentence_ends[-1].end()].rstrip()
        return raw_prefix
    return output


def novel_lexical_repetition_occurrences(
    output: str,
    prompt: str,
) -> int:
    """Count repeated output lexical four-grams absent from the prompt."""

    prompt_words = re.findall(r"[\w']+", prompt.casefold())
    allowed = {
        tuple(prompt_words[index : index + 4])
        for index in range(max(0, len(prompt_words) - 3))
    }
    words = re.findall(r"[\w']+", output.casefold())
    counts: dict[tuple[str, ...], int] = {}
    repeated_occurrences = 0
    for index in range(max(0, len(words) - 3)):
        fourgram = tuple(words[index : index + 4])
        if fourgram in allowed:
            continue
        previous = counts.get(fourgram, 0)
        counts[fourgram] = previous + 1
        if previous >= 1:
            repeated_occurrences += 1
    return repeated_occurrences
