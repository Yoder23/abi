"""Data, labeling, and independent scoring for R21."""

from __future__ import annotations

import collections
import hashlib
import math
import random
import re
from typing import Any, Iterable

from experiments.instructional_realization_r20.protocol import (
    EXTRACTION_INSTRUCTIONS,
    SLOT_KEYS,
    TASKS,
    _slots,
    public_rows,
    render_prompt,
)

FORMAT = "abi-r21-label-separated-generative-transfer/1"
TRAIN_ROWS_PER_TASK = 100
TRAIN_OFFSET = 1000
SEEDS = (21021, 21022, 21023)
LABELS = tuple(TASKS)
TOKEN_RE = re.compile(r"[a-z]+")
NUMBER_RE = re.compile(r"\d+")
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and as at be before by for from in is it of on or the these this to was were with".split()
)


def training_rows() -> list[dict[str, Any]]:
    rows = []
    for task_index, task in enumerate(TASKS):
        bank = EXTRACTION_INSTRUCTIONS[task]
        for repeat in range(TRAIN_ROWS_PER_TASK):
            index = TRAIN_OFFSET + task_index * TRAIN_ROWS_PER_TASK + repeat
            slots = _slots(task, index)
            instruction = bank[repeat % len(bank)]
            prompt = render_prompt(instruction, slots)
            identity = hashlib.sha256(
                f"r21|train|{task}|{repeat}|{prompt}".encode()
            ).hexdigest()[:20]
            rows.append(
                {
                    "record_id": f"r21-public-{identity}",
                    "split": "training",
                    "task": task,
                    "instruction": instruction,
                    "slots": slots,
                    "prompt": prompt,
                }
            )
    return rows


def evaluation_rows() -> list[dict[str, Any]]:
    return public_rows("evaluation")


def normalized_prompt(label: str, slots: dict[str, str]) -> str:
    if label not in LABELS or set(slots) != set(SLOT_KEYS):
        raise ValueError("R21 normalized prompt is invalid")
    return "\n".join(
        [f"TASK_LABEL: {label}", "DATA:", *(f"{key}={slots[key]}" for key in SLOT_KEYS)]
    )


def instruction_from_prompt(prompt: str) -> str:
    first = prompt.splitlines()[0] if prompt else ""
    if not first.startswith("INSTRUCTION: "):
        raise ValueError("R21 instruction boundary changed")
    return first.removeprefix("INSTRUCTION: ")


class WordLabeler:
    """Small transparent labeler fit only from imported teacher labels."""

    def __init__(self, document: dict[str, Any]):
        if (
            document.get("format") != "abi-r21-word-labeler/1"
            or tuple(document.get("labels", ())) != LABELS
        ):
            raise ValueError("R21 labeler document changed")
        self.document = document

    @classmethod
    def fit(cls, rows: Iterable[dict[str, str]]) -> "WordLabeler":
        counts = {label: collections.Counter() for label in LABELS}
        totals = collections.Counter()
        vocabulary: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for row in rows:
            label = str(row["label"])
            if label not in LABELS:
                raise ValueError("R21 teacher label is outside ontology")
            tokens = TOKEN_RE.findall(str(row["instruction"]).lower())
            if not tokens:
                raise ValueError("R21 empty labeling instruction")
            pair = (label, str(row["instruction"]).lower())
            if pair not in seen_pairs:
                counts[label].update(tokens)
                totals[label] += len(tokens)
                vocabulary.update(tokens)
                seen_pairs.add(pair)
        if len(seen_pairs) != len(LABELS) * 6 or any(not counts[x] for x in LABELS):
            raise ValueError("R21 label training coverage is incomplete")
        document = {
            "format": "abi-r21-word-labeler/1",
            "labels": list(LABELS),
            "vocabulary": sorted(vocabulary),
            "counts": {label: dict(sorted(counts[label].items())) for label in LABELS},
            "totals": {label: totals[label] for label in LABELS},
            "smoothing": 1.0,
        }
        document["sha256"] = hashlib.sha256(
            canonical_bytes(document)
        ).hexdigest()
        return cls(document)

    def predict(self, instruction: str) -> str:
        tokens = TOKEN_RE.findall(instruction.lower())
        vocabulary = self.document["vocabulary"]
        size = len(vocabulary)
        scores = {}
        for label in LABELS:
            counts = self.document["counts"][label]
            total = int(self.document["totals"][label])
            scores[label] = sum(
                math.log((int(counts.get(token, 0)) + 1.0) / (total + size))
                for token in tokens
            )
        return max(LABELS, key=lambda label: (scores[label], -LABELS.index(label)))


def canonical_bytes(value: Any) -> bytes:
    import json

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _tokens(value: str) -> list[str]:
    return WORD_RE.findall(value.lower())


def _important(value: str) -> set[str]:
    return {token for token in _tokens(value) if token not in STOPWORDS and len(token) > 1}


def _lcs(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for item in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if item == other
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def lexical_f1(output: str, reference: str) -> float:
    left, right = _tokens(output), _tokens(reference)
    if not left or not right:
        return float(left == right)
    common = collections.Counter(left) & collections.Counter(right)
    overlap = sum(common.values())
    precision, recall = overlap / len(left), overlap / len(right)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def score_output(row: dict[str, Any], output: str, teacher_output: str) -> dict[str, Any]:
    task = str(row["task"])
    lowered = output.lower()
    fields = [str(row["slots"][key]) for key in SLOT_KEYS]
    field_recalls = []
    for field in fields:
        wanted = _important(field)
        field_recalls.append(
            len(wanted & set(_tokens(output))) / len(wanted) if wanted else 1.0
        )
    prompt_numbers = set(NUMBER_RE.findall(str(row["prompt"])))
    output_numbers = set(NUMBER_RE.findall(output))
    hallucination_pass = output_numbers <= prompt_numbers
    if task == "abstention":
        semantic_pass = any(
            phrase in lowered
            for phrase in (
                "abstain", "cannot", "can't", "unable", "unavailable",
                "not provided", "not supplied", "unsupported", "decline",
            )
        )
    else:
        minimum = 0.75 if task in {"prose", "bullets"} else 0.50
        semantic_pass = all(value >= minimum for value in field_recalls)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if task == "prose":
        adherence = len(lines) == 1 and output.rstrip().endswith((".", "!", "?"))
    elif task == "summary":
        adherence = bool(lines) and len(output.encode()) <= 700
    elif task == "email":
        adherence = (
            "subject" in lowered
            and any(value in lowered for value in ("dear", "hello", "message:"))
            and len(lines) >= 3
        )
    elif task == "bullets":
        adherence = sum(line.startswith(("-", "*", "•")) for line in lines) == 3
    elif task == "clarification":
        adherence = output.rstrip().endswith("?") and any(
            value in lowered for value in ("what", "which", "could", "clarif")
        )
    else:
        adherence = semantic_pass and len(output.encode()) <= 500
    words = _tokens(output)
    trigrams = collections.Counter(tuple(words[i : i + 3]) for i in range(max(0, len(words) - 2)))
    no_collapse = (
        bool(words)
        and len(output.encode()) <= 1000
        and max(trigrams.values(), default=0) <= 3
        and max(collections.Counter(words).values(), default=0) <= max(6, len(words) // 3)
    )
    fluency_pass = no_collapse and sum(char.isalpha() for char in output) >= 3
    functional = semantic_pass and adherence and hallucination_pass and fluency_pass
    teacher_tokens, output_tokens = _tokens(teacher_output), _tokens(output)
    lcs = _lcs(output_tokens, teacher_tokens)
    rouge_l = (
        2 * (lcs / len(output_tokens)) * (lcs / len(teacher_tokens))
        / ((lcs / len(output_tokens)) + (lcs / len(teacher_tokens)))
        if lcs and output_tokens and teacher_tokens
        else 0.0
    )
    return {
        "field_recalls": field_recalls,
        "semantic_pass": semantic_pass,
        "adherence_pass": adherence,
        "hallucination_pass": hallucination_pass,
        "fluency_pass": fluency_pass,
        "repetition_collapse": not no_collapse,
        "functional_pass": functional,
        "teacher_lexical_f1": lexical_f1(output, teacher_output),
        "teacher_rouge_l": rouge_l,
        "output_utf8_bytes": len(output.encode()),
    }


def bootstrap_difference(
    left: list[float], right: list[float], *, seed: int, samples: int = 10_000
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap inputs differ")
    randomizer = random.Random(seed)
    deltas = []
    for _ in range(samples):
        values = [left[index] - right[index] for index in (randomizer.randrange(len(left)) for _ in left)]
        deltas.append(sum(values) / len(values))
    deltas.sort()
    return {
        "mean_difference": sum(a - b for a, b in zip(left, right)) / len(left),
        "ci95_low": deltas[int(0.025 * samples)],
        "ci95_high": deltas[int(0.975 * samples) - 1],
        "bootstrap_samples": samples,
    }
