"""Frozen rows and public-derived semantic contract for R23."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from experiments.generative_transfer_r21.protocol import score_output
from experiments.instructional_realization_r20.protocol import (
    PROGRAMS,
    SLOT_KEYS,
    TASKS,
    _slots,
    render_prompt,
)

HIDDEN_INSTRUCTIONS = {
    "prose": (
        "Express the supplied material as one grammatical prose sentence.",
        "Turn every supplied value into a single ordinary sentence.",
        "Write one fluent sentence containing the complete supplied content.",
        "Combine these three values into one plain prose statement.",
    ),
    "summary": (
        "Give one concise summary that retains every supplied point.",
        "Compress the supplied information into a brief summary line.",
        "Write a compact summary of all three supplied facts.",
        "Produce a concise summary of these data without adding information.",
    ),
    "email": (
        "Turn these notes into a short professional email.",
        "Write a concise email containing each supplied value.",
        "Produce a professional email with subject, message, and sender.",
        "Draft an email note from all of the supplied content.",
    ),
    "bullets": (
        "Return exactly three bullets, one for each supplied value.",
        "Arrange all supplied content as a three-item bullet list.",
        "Put each of these values on its own bullet line.",
        "Present the three supplied points in bullet form.",
    ),
    "clarification": (
        "Ask a single clarification question about all supplied values.",
        "Ask one concise clarification question about this supplied situation.",
        "Request clarification of the missing detail in one question.",
        "Write one question that clarifies the supplied information.",
    ),
    "abstention": (
        "Abstain safely instead of inventing the unavailable detail.",
        "Refuse to fabricate information not established by these data.",
        "State that the unsupported detail cannot be provided.",
        "Decline this request because the requested fact is unavailable.",
    ),
}

SUMMARY_MARKERS = {
    "stable": (
        "stable",
        "stability",
        "steady",
        "no change",
        "no significant change",
    ),
    "decrease": ("fell", "decrease", "decreased", "reduction", "reduced", "improved"),
    "review": ("review",),
}
NUMBER_RE = re.compile(r"\d+")
CASE_RE = re.compile(r"case-(\d+)")
MILLISECOND_RE = re.compile(r"(\d+) milliseconds?")
DAY_RE = re.compile(r"day (\d+)")


def _one(pattern: re.Pattern[str], value: str, name: str) -> str:
    matches = pattern.findall(value.lower())
    if len(matches) != 1:
        raise ValueError(f"R23 expected one {name} value")
    return matches[0]


def _marker_present(value: str, marker: str) -> bool:
    return bool(re.search(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", value))


def seed_commitment(seed_hex: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", seed_hex):
        raise ValueError("R23 hidden seed must be 256-bit lowercase hex")
    return hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest()


def hidden_rows(seed_hex: str) -> list[dict[str, Any]]:
    seed_commitment(seed_hex)
    randomizer = random.Random(int(seed_hex, 16))
    indices = randomizer.sample(range(1_000_000, 9_000_000), len(TASKS) * 20)
    rows = []
    position = 0
    for task in TASKS:
        instructions = list(HIDDEN_INSTRUCTIONS[task]) * 5
        randomizer.shuffle(instructions)
        for repeat, instruction in enumerate(instructions):
            index = indices[position]
            position += 1
            slots = _slots(task, index)
            if set(slots) != set(SLOT_KEYS):
                raise ValueError("R23 hidden slot schema changed")
            prompt = render_prompt(instruction, slots)
            identity = hashlib.sha256(
                f"r23|hidden|{task}|{repeat}|{prompt}".encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "record_id": f"r23-hidden-{identity}",
                    "split": "fresh_semantic_replication",
                    "task": task,
                    "instruction": instruction,
                    "slots": slots,
                    "prompt": prompt,
                    "expected": PROGRAMS[task].format(**slots),
                }
            )
    if len(rows) != 120 or len({row["record_id"] for row in rows}) != 120:
        raise ValueError("R23 hidden matrix changed")
    return rows


def semantic_score(
    row: dict[str, Any], output: str, teacher_output: str
) -> dict[str, Any]:
    result = score_output(row, output, teacher_output)
    if row["task"] != "summary":
        return result
    lowered = output.lower()
    prompt_numbers = set(NUMBER_RE.findall(str(row["prompt"])))
    output_numbers = set(NUMBER_RE.findall(output))
    case_id = _one(CASE_RE, str(row["slots"]["field_a"]), "case")
    latency = _one(MILLISECOND_RE, str(row["slots"]["field_b"]), "latency")
    day = _one(DAY_RE, str(row["slots"]["field_c"]), "day")
    stable = any(
        _marker_present(lowered, marker) for marker in SUMMARY_MARKERS["stable"]
    )
    decrease = any(
        _marker_present(lowered, marker) for marker in SUMMARY_MARKERS["decrease"]
    )
    assertions = {
        "all_prompt_numbers_present": prompt_numbers <= output_numbers,
        "no_new_numbers": output_numbers <= prompt_numbers,
        "case_identity_bound": f"case-{case_id}" in lowered,
        "stable_concept_present": stable and "unstable" not in lowered,
        "latency_amount_bound": bool(
            re.search(
                rf"(?:latency[^.\n;]{{0,80}}{re.escape(latency)} milliseconds?"
                rf"|{re.escape(latency)} milliseconds?[^.\n;]{{0,80}}latency)",
                lowered,
            )
        ),
        "decrease_concept_present": decrease
        and not any(
            word in lowered
            for word in ("increase", "increased", "rose", "worsened")
        ),
        "review_day_bound": bool(
            re.search(rf"review[^.\n;]{{0,100}}day {re.escape(day)}", lowered)
        ),
        "review_not_cancelled": not any(
            word in lowered for word in ("cancelled", "canceled", "unscheduled")
        ),
    }
    semantic = all(assertions.values())
    result["semantic_contract"] = assertions
    result["semantic_pass"] = semantic
    result["functional_pass"] = bool(
        semantic
        and result["adherence_pass"]
        and result["hallucination_pass"]
        and result["fluency_pass"]
    )
    return result


def validate_public_lexicon(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Prove each equivalence marker appears in frozen public prompts/outputs."""
    corpus = "\n".join(
        f"{row.get('prompt', '')}\n{row.get('teacher_output', '')}".lower()
        for row in rows
        if row.get("task") == "summary"
    )
    counts = {
        marker: corpus.count(marker)
        for markers in SUMMARY_MARKERS.values()
        for marker in markers
    }
    if not counts or any(count < 1 for count in counts.values()):
        raise ValueError("R23 semantic marker lacks public-corpus support")
    return counts
