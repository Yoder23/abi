"""Committed hidden-row generator for R21 replication."""

from __future__ import annotations

import hashlib
import random
import re
from typing import Any

from experiments.instructional_realization_r20.protocol import (
    PROGRAMS,
    SLOT_KEYS,
    TASKS,
    _slots,
    render_prompt,
)

HIDDEN_INSTRUCTIONS = {
    "prose": (
        "Compose a plain prose sentence using the supplied fields.",
        "Write one prose sentence that preserves the supplied content.",
        "Render the three data fields as one coherent sentence.",
        "Produce a single plain sentence from all supplied values.",
    ),
    "summary": (
        "Condense the supplied fields into one concise summary.",
        "Create a brief labeled summary using all three points.",
        "Summarize every supplied value in one compact line.",
        "Provide one concise summary without inventing information.",
    ),
    "email": (
        "Prepare a professional email message from the supplied fields.",
        "Draft a concise email with subject, body, and sender.",
        "Compose a short email note using every supplied value.",
        "Create a compact professional email from these data.",
    ),
    "bullets": (
        "Itemize the supplied fields as exactly three bullet points.",
        "Format every supplied value on its own bullet line.",
        "Present all three data fields in a bulleted list.",
        "Create three bullet items using only the supplied content.",
    ),
    "clarification": (
        "Form one concise clarification question about the supplied fields.",
        "Ask one question that clarifies all three data values.",
        "Request the missing detail in one clarification question.",
        "Use every supplied field to ask what needs clarifying.",
    ),
    "abstention": (
        "Give a safe abstention and do not invent unavailable information.",
        "Decline the unsupported request rather than adding a fact.",
        "State that the requested detail cannot be supplied from these data.",
        "Refuse to fabricate the missing information in the supplied fields.",
    ),
}


def seed_commitment(seed_hex: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", seed_hex):
        raise ValueError("R21 hidden seed must be 256-bit lowercase hex")
    return hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest()


def hidden_rows(seed_hex: str) -> list[dict[str, Any]]:
    seed_commitment(seed_hex)
    randomizer = random.Random(int(seed_hex, 16))
    indices = randomizer.sample(range(50_000, 950_000), len(TASKS) * 20)
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
                raise ValueError("R21 hidden slot schema changed")
            prompt = render_prompt(instruction, slots)
            identity = hashlib.sha256(f"r21|hidden|{task}|{repeat}|{prompt}".encode()).hexdigest()[
                :20
            ]
            rows.append(
                {
                    "record_id": f"r21-hidden-{identity}",
                    "split": "hidden_replication",
                    "task": task,
                    "instruction": instruction,
                    "slots": slots,
                    "prompt": prompt,
                    "expected": PROGRAMS[task].format(**slots),
                }
            )
    if len(rows) != 120 or len({row["record_id"] for row in rows}) != 120:
        raise ValueError("R21 hidden matrix changed")
    return rows
