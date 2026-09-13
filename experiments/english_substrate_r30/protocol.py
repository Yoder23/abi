"""Prospectively defined tasks and scorers for the R30 public pilot."""

from __future__ import annotations

import collections
import hashlib
import random
import re
from typing import Any


TASKS = (
    "prose",
    "summary",
    "rewrite",
    "email",
    "bullets",
    "tone",
    "clarification",
    "conversation",
    "planning",
    "comparison",
    "reasoning",
    "abstention",
)
TRAIN_PER_TASK = 24
EVAL_PER_TASK = 12
WORD_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\d+")

INSTRUCTIONS = {
    "prose": (
        "Turn the supplied notes into one grammatical sentence.",
        "Express the supplied material as a single prose sentence.",
        "Combine the provided details into one natural sentence.",
    ),
    "summary": (
        "Summarize the supplied passage in one concise paragraph.",
        "Write a compact summary retaining every important supplied point.",
        "Condense the supplied material without adding information.",
    ),
    "rewrite": (
        "Rewrite the supplied sentence for clarity without changing its meaning.",
        "Paraphrase the supplied text in clear natural English.",
        "Improve the wording while preserving the complete supplied meaning.",
    ),
    "email": (
        "Draft a short professional email from the supplied notes.",
        "Turn the supplied material into a concise email with a subject and signoff.",
        "Write a professional email that includes every supplied detail.",
    ),
    "bullets": (
        "Present the supplied items as exactly three bullet points.",
        "Convert the supplied material into a three-item bullet list.",
        "Return one bullet for each of the three supplied details.",
    ),
    "tone": (
        "Rewrite the supplied message in the requested tone while preserving meaning.",
        "Express the supplied content using the specified tone.",
        "Adjust the wording to the requested tone without adding facts.",
    ),
    "clarification": (
        "Ask one concise question that clarifies the missing supplied detail.",
        "Request the one missing detail with a clear question.",
        "Write one clarification question grounded in the supplied situation.",
    ),
    "conversation": (
        "Reply conversationally and helpfully using only the supplied situation.",
        "Write a brief grounded conversational response to the supplied message.",
        "Respond naturally to the message without inventing background facts.",
    ),
    "planning": (
        "Create a three-step plan from the supplied goal, resources, and constraint.",
        "Turn the supplied material into exactly three practical ordered steps.",
        "Write a concise three-step plan using only the supplied information.",
    ),
    "comparison": (
        "Compare the two supplied options using only the supplied criterion.",
        "Write a concise comparison grounded in the provided options and measure.",
        "Contrast both supplied choices without introducing outside facts.",
    ),
    "reasoning": (
        "State the conclusion implied by the supplied fictional rule and observations.",
        "Reason from the supplied invented rule and give only the grounded conclusion.",
        "Explain the conclusion supported by the supplied synthetic premises.",
    ),
    "abstention": (
        "Decline to answer because the requested fact is not in the supplied material.",
        "State that the unsupported fact cannot be provided from the supplied context.",
        "Abstain rather than using outside knowledge or inventing an answer.",
    ),
}

VERBS = ("reviewed", "measured", "organized", "checked", "recorded", "scheduled")
OBJECTS = ("sample", "report", "packet", "instrument", "draft", "request")
TONES = ("warm", "formal", "calm", "direct")
CRITERIA = ("completion time", "error count", "review effort", "queue length")


def _details(task: str, index: int) -> list[str]:
    nonce = f"Virelon{index:05d}"
    verb = VERBS[index % len(VERBS)]
    obj = OBJECTS[(index * 5 + 1) % len(OBJECTS)]
    amount = 1000 + index * 7
    if task in {"prose", "summary", "rewrite"}:
        return [
            f"identifier={nonce}",
            f"event=the coordinator {verb} the {obj}",
            f"result=the recorded value was {amount}",
            f"next=review occurs on day {index % 27 + 1}",
        ]
    if task == "email":
        return [
            f"identifier={nonce}",
            f"recipient=the review team",
            f"request=please {verb} {obj} {amount}",
            f"sender=Coordinator {index % 41 + 1}",
        ]
    if task == "bullets":
        return [
            f"first={nonce}",
            f"second={verb} {obj} {amount}",
            f"third=archive on day {index % 27 + 1}",
        ]
    if task == "tone":
        return [
            f"identifier={nonce}",
            f"tone={TONES[index % len(TONES)]}",
            f"message=please {verb} the {obj} numbered {amount}",
        ]
    if task == "clarification":
        return [
            f"identifier={nonce}",
            f"known=team {index % 9 + 1} will {verb} the {obj}",
            f"missing=the completion date for item {amount}",
        ]
    if task == "conversation":
        return [
            f"identifier={nonce}",
            f"message=I finished the {obj} but the value {amount} surprised me",
            f"preference=respond briefly and suggest checking the recorded notes",
        ]
    if task == "planning":
        return [
            f"identifier={nonce}",
            f"goal={verb} the {obj} numbered {amount}",
            f"resource=one checklist and team {index % 9 + 1}",
            f"constraint=finish before day {index % 27 + 1}",
        ]
    if task == "comparison":
        return [
            f"identifier={nonce}",
            f"option_a=method A recorded {amount}",
            f"option_b=method B recorded {amount + 3}",
            f"criterion={CRITERIA[index % len(CRITERIA)]} where lower is preferred",
        ]
    if task == "reasoning":
        return [
            f"identifier={nonce}",
            "rule=every glim is checked before any nork",
            f"observation={nonce} is a glim and item {amount} is a nork",
            "question=which one must be checked first",
        ]
    if task == "abstention":
        external = (
            "the national independence day",
            "the chemical boiling point",
            "the result of an undocumented Python program",
        )[index % 3]
        return [
            f"identifier={nonce}",
            f"request={external} for {nonce}",
            "supplied_context=no answer or supporting source is provided",
        ]
    raise ValueError(f"unknown R30 task: {task}")


def rows(split: str, *, seed: int = 30030) -> list[dict[str, Any]]:
    if split not in {"train", "evaluation"}:
        raise ValueError("R30 split must be train or evaluation")
    per_task = TRAIN_PER_TASK if split == "train" else EVAL_PER_TASK
    offset = 0 if split == "train" else 100_000
    generated = []
    for task_index, task in enumerate(TASKS):
        bank = list(INSTRUCTIONS[task])
        random.Random(seed + task_index + (0 if split == "train" else 10_000)).shuffle(bank)
        for repeat in range(per_task):
            index = offset + task_index * per_task + repeat
            instruction = bank[repeat % len(bank)]
            details = _details(task, index)
            prompt = "\n".join((f"INSTRUCTION: {instruction}", "SUPPLIED MATERIAL:", *details))
            digest = hashlib.sha256(f"r30|{split}|{task}|{repeat}|{prompt}".encode()).hexdigest()[:20]
            generated.append(
                {
                    "record_id": f"r30-{split}-{digest}",
                    "split": split,
                    "oracle_task": task,
                    "instruction": instruction,
                    "details": details,
                    "nonce": f"Virelon{index:05d}",
                    "prompt": prompt,
                }
            )
    return generated


def normalize_free_label(value: str) -> str | None:
    label = re.sub(r"[^a-z ]+", " ", value.casefold()).strip()
    rules = (
        ("abstention", ("abstain", "refusal", "declin", "unknown", "insufficient")),
        ("clarification", ("clarif", "question asking", "information request")),
        ("email", ("email", "correspondence")),
        ("bullets", ("bullet", "list formatting", "itemization")),
        ("summary", ("summar", "condens")),
        ("rewrite", ("rewrit", "paraphras", "rephrasing", "editing")),
        ("tone", ("tone", "style transfer", "style adaptation")),
        ("conversation", ("conversation", "dialogue", "empathetic", "chat")),
        ("planning", ("plan", "procedural", "steps")),
        ("comparison", ("compar", "contrast")),
        ("reasoning", ("reason", "logic", "inference", "deduction")),
        ("prose", ("prose", "sentence generation", "sentence construction", "writing")),
    )
    matches = [canonical for canonical, needles in rules if any(item in label for item in needles)]
    return matches[0] if len(set(matches)) == 1 else None


def _tokens(value: str) -> list[str]:
    return WORD_RE.findall(value.casefold())


def lexical_f1(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return float(a == b)
    overlap = sum((collections.Counter(a) & collections.Counter(b)).values())
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score(row: dict[str, Any], output: str, teacher: str) -> dict[str, Any]:
    task = str(row["oracle_task"])
    lowered = output.casefold()
    prompt_numbers = set(NUMBER_RE.findall(str(row["prompt"])))
    output_numbers = set(NUMBER_RE.findall(output))
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    words = _tokens(output)
    trigrams = collections.Counter(tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2)))
    grounded = str(row["nonce"]).casefold() in lowered and output_numbers <= prompt_numbers
    noncollapsed = bool(words) and len(output.encode()) <= 1400 and max(trigrams.values(), default=0) <= 3
    if task == "prose":
        adherence = len(lines) == 1 and output.rstrip().endswith((".", "!", "?"))
    elif task in {"summary", "rewrite"}:
        adherence = 1 <= len(lines) <= 3
    elif task == "email":
        adherence = "subject" in lowered and len(lines) >= 3
    elif task == "bullets":
        adherence = sum(line.startswith(("-", "*", "•")) for line in lines) == 3
    elif task == "tone":
        tone = next(item.split("=", 1)[1] for item in row["details"] if item.startswith("tone="))
        adherence = tone in lowered or len(lines) <= 3
    elif task == "clarification":
        adherence = output.rstrip().endswith("?")
    elif task == "planning":
        adherence = sum(line.startswith(("1", "2", "3", "-", "*")) for line in lines) >= 3
    elif task == "comparison":
        adherence = all(marker in lowered for marker in ("method a", "method b"))
    elif task == "reasoning":
        adherence = "glim" in lowered and "nork" in lowered
    elif task == "abstention":
        adherence = any(marker in lowered for marker in ("cannot", "can't", "not provided", "insufficient", "unable", "no answer"))
        grounded = output_numbers <= prompt_numbers
    else:
        adherence = 1 <= len(lines) <= 4
    return {
        "grounded": grounded,
        "adherence": adherence,
        "noncollapsed": noncollapsed,
        "functional": bool(grounded and adherence and noncollapsed),
        "teacher_lexical_f1": lexical_f1(output, teacher),
        "output_bytes": len(output.encode()),
    }

