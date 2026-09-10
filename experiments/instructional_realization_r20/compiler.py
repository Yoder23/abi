"""Zero-training R20 teacher-output program and instruction compiler."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .package import PACKAGE_FORMAT, R20PackageError, features, parse_prompt


class R20CompilerError(RuntimeError):
    """Raised when teacher evidence cannot identify the R20 package."""


def parse_program(output: str, slots: dict[str, str]) -> str:
    program = output.strip().replace("\r\n", "\n")
    for key, value in sorted(slots.items(), key=lambda item: -len(item[1])):
        if program.count(value) != 1:
            raise R20CompilerError("teacher output omitted or duplicated supplied content")
        program = program.replace(value, f"{{{key}}}")
    return program


def infer_package(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(records) != 72 or len({record.get("record_id") for record in records}) != 72:
        raise R20CompilerError("R20 extraction row inventory changed")
    parsed: list[tuple[str, str]] = []
    rejected = 0
    for record in records:
        if set(record) != {"record_id", "prompt", "output"}:
            raise R20CompilerError("R20 extraction record schema changed")
        try:
            instruction, slots = parse_prompt(str(record["prompt"]))
            program = parse_program(str(record["output"]), slots)
        except (R20CompilerError, R20PackageError, ValueError):
            rejected += 1
            continue
        parsed.append((instruction, program))
    support = Counter(program for _, program in parsed)
    selected = [program for program, count in support.most_common(6) if count >= 6]
    if len(selected) != 6:
        raise R20CompilerError("insufficient teacher support for six output programs")
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for instruction, program in parsed:
        if program not in selected:
            continue
        grouped[program].update(features(instruction))
    classes = [
        {
            "program": program,
            "support": support[program],
            "centroid": dict(sorted(grouped[program].items())),
        }
        for program in sorted(selected)
    ]
    package = {
        "format": PACKAGE_FORMAT,
        "namespace": "english/core/instructional-realization",
        "representation": "instruction-ngram-centroid-plus-output-program",
        "classes": classes,
    }
    diagnostics = {
        "records_consumed": len(records),
        "records_parsed": len(parsed),
        "records_rejected": rejected,
        "programs_selected": len(classes),
        "selected_support": {program: support[program] for program in sorted(selected)},
        "task_labels_consumed": 0,
        "expected_outputs_consumed": 0,
        "evaluation_rows_consumed": 0,
        "training_steps": 0,
    }
    return package, diagnostics
