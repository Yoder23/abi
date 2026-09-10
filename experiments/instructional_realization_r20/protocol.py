"""Registered R20 prompts, splits, and independent functional oracle."""

from __future__ import annotations

import hashlib
from typing import Any

TASKS = ("prose", "summary", "email", "bullets", "clarification", "abstention")
SLOT_KEYS = ("field_a", "field_b", "field_c")

EXTRACTION_INSTRUCTIONS = {
    "prose": (
        "Compose one sentence from the supplied content.",
        "Write the three fields as a single sentence.",
        "Render this content as one prose sentence.",
        "Turn the supplied fields into a sentence.",
        "Produce one plain sentence using the data.",
        "Express the content in a single prose line.",
    ),
    "summary": (
        "Summarize the supplied points concisely.",
        "Write a concise summary of the three fields.",
        "Condense this content into a labeled summary.",
        "Provide a brief summary using only the data.",
        "Create a compact summary of these points.",
        "Reduce the supplied content to one summary line.",
    ),
    "email": (
        "Draft a short email from the supplied content.",
        "Write a compact email using these fields.",
        "Turn the data into a professional email.",
        "Compose an email with subject, body, and sender.",
        "Prepare a concise email from these notes.",
        "Create an email message using only the data.",
    ),
    "bullets": (
        "Format the supplied content as three bullet points.",
        "Turn these fields into a bulleted list.",
        "Present the data as three bullets.",
        "List each supplied field as a bullet item.",
        "Create a three-item bullet list from the content.",
        "Itemize the supplied points with bullets.",
    ),
    "clarification": (
        "Ask one clarification question using the supplied content.",
        "Request clarification about these fields.",
        "Write a concise question that asks for clarification.",
        "Use the data to ask what needs clarifying.",
        "Form one clarification question from the content.",
        "Ask for the missing detail described by these fields.",
    ),
    "abstention": (
        "Decline the unsupported request using the supplied content.",
        "Refuse to invent the requested fact.",
        "State that the unsupported information was not supplied.",
        "Give a concise abstention instead of adding a fact.",
        "Do not invent the requested detail; decline it.",
        "Respond safely that the requested information is unavailable.",
    ),
}

EVALUATION_INSTRUCTIONS = {
    "prose": (
        "Create a single sentence from these data fields.",
        "Use the content to produce one coherent sentence.",
        "Combine all three values into one prose sentence.",
        "State the supplied content in a single sentence.",
        "Make one sentence that preserves every field.",
    ),
    "summary": (
        "Give a short labeled summary of the supplied content.",
        "Summarize all three points in one compact line.",
        "Produce a concise summary without adding information.",
        "Condense the data into a single summary line.",
        "Create a brief summary that preserves each point.",
    ),
    "email": (
        "Generate a brief professional email from these data.",
        "Use all three fields to compose a short email.",
        "Prepare an email with a subject, message, and signoff.",
        "Turn the supplied values into an email note.",
        "Write the requested compact email without extra facts.",
    ),
    "bullets": (
        "Create a bullet list containing the three supplied values.",
        "Display each field on its own bullet line.",
        "Convert the content into exactly three bullets.",
        "Itemize all supplied values as bullet points.",
        "Return a three-line bulleted list from the data.",
    ),
    "clarification": (
        "Pose one concise clarification question from these data.",
        "Ask for clarification using every supplied field.",
        "Turn the content into a question requesting clarification.",
        "Request the described missing detail with one question.",
        "Write one question that clarifies the supplied situation.",
    ),
    "abstention": (
        "Abstain because the requested information was not provided.",
        "Decline rather than fabricate the unsupported detail.",
        "Say that the requested fact cannot be supplied from these data.",
        "Refuse the unsupported request without inventing information.",
        "Provide a safe abstention for the missing fact.",
    ),
}

PROGRAMS = {
    "prose": "{field_a} {field_b} {field_c}.",
    "summary": "Summary: {field_a}; {field_b}; {field_c}.",
    "email": "Subject: {field_a}\n\nHello,\n\n{field_b}\n\nRegards,\n{field_c}",
    "bullets": "- {field_a}\n- {field_b}\n- {field_c}",
    "clarification": "Could you clarify {field_a} for {field_b} before {field_c}?",
    "abstention": "I cannot provide {field_a} about {field_b} because {field_c} was not supplied.",
}


def _slots(task: str, index: int) -> dict[str, str]:
    code = f"case-{index:03d}"
    if task == "prose":
        return {
            "field_a": f"Analyst {code}",
            "field_b": "reviewed the stability report for",
            "field_c": f"instrument {index + 17}",
        }
    if task == "summary":
        return {
            "field_a": f"sensor {code} remained stable",
            "field_b": f"latency fell by {index % 9 + 1} milliseconds",
            "field_c": f"review is scheduled for day {index % 27 + 1}",
        }
    if task == "email":
        return {
            "field_a": f"Review request {code}",
            "field_b": f"Please review the attached notes for item {index + 31}.",
            "field_c": f"Coordinator {index + 5}",
        }
    if task == "bullets":
        return {
            "field_a": f"verify sample {index + 2}",
            "field_b": f"record batch {index + 11}",
            "field_c": f"archive report {index + 23}",
        }
    if task == "clarification":
        return {
            "field_a": f"the deadline for {code}",
            "field_b": f"team {index % 7 + 1}",
            "field_c": f"checkpoint {index + 41}",
        }
    if task == "abstention":
        return {
            "field_a": f"the unlisted rate for {code}",
            "field_b": f"region {index % 8 + 1}",
            "field_c": f"source note {index + 53}",
        }
    raise ValueError("unknown R20 task")


def render_prompt(instruction: str, slots: dict[str, str]) -> str:
    if set(slots) != set(SLOT_KEYS):
        raise ValueError("R20 slot schema changed")
    return "\n".join(
        [
            f"INSTRUCTION: {instruction}",
            "DATA:",
            *(f"{key}={slots[key]}" for key in SLOT_KEYS),
        ]
    )


def _fill(program: str, slots: dict[str, str]) -> str:
    return program.format(**slots)


def public_rows(split: str) -> list[dict[str, Any]]:
    if split not in {"extraction", "evaluation"}:
        raise ValueError("unknown R20 public split")
    repeats = 12 if split == "extraction" else 20
    offset = 0 if split == "extraction" else 100
    instruction_bank = EXTRACTION_INSTRUCTIONS if split == "extraction" else EVALUATION_INSTRUCTIONS
    rows = []
    for task_index, task in enumerate(TASKS):
        instructions = instruction_bank[task]
        for repeat in range(repeats):
            index = offset + task_index * repeats + repeat
            slots = _slots(task, index)
            instruction = instructions[repeat % len(instructions)]
            prompt = render_prompt(instruction, slots)
            identity = hashlib.sha256(f"{split}|{task}|{repeat}|{prompt}".encode()).hexdigest()[:20]
            rows.append(
                {
                    "record_id": f"r20-public-{identity}",
                    "split": split,
                    "task": task,
                    "instruction": instruction,
                    "slots": slots,
                    "prompt": prompt,
                    "expected": _fill(PROGRAMS[task], slots),
                }
            )
    return rows


def replace_instruction(prompt: str, instruction: str) -> str:
    lines = prompt.splitlines()
    if not lines or not lines[0].startswith("INSTRUCTION: "):
        raise ValueError("R20 prompt schema changed")
    return "\n".join([f"INSTRUCTION: {instruction}", *lines[1:]])
