"""Frozen R24 data preparation and scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error

NAMESPACES = ("chemistry/periodic-table", "geography/national-capitals")
SEEDS = (24021, 24022, 24023)


def jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R24 JSONL unreadable: {path}") from exc


def source_splits(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = jsonl(path)
    training = [row for row in rows if row.get("split") == "extraction"]
    evaluation = [row for row in rows if row.get("split") == "evaluation"]
    if (
        len(rows) != 96
        or len(training) != 48
        or len(evaluation) != 48
        or {row.get("namespace") for row in rows} != set(NAMESPACES)
        or any(row.get("completion_exact") is not True for row in rows)
    ):
        raise R14Error("R24 R16 source split changed")
    return training, evaluation


def prepared_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        question = str(row["question"])
        terminal = question[-1]
        if terminal not in {".", "?"} or question.count(terminal) != 1:
            raise R14Error("R24 question has no safe tokenizer control lexeme")
        prepared.append(
            {
                **row,
                "record_id": f"{row['fact_id']}-view{row['view']}",
                "prompt": question,
                "response": str(row["answer"]),
                "copy_lexemes": [terminal],
            }
        )
    return prepared


def domain_slug(namespace: str) -> str:
    if namespace not in NAMESPACES:
        raise R14Error("R24 unknown namespace")
    return namespace.replace("/", "-")
