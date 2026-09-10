"""Compact R20 instruction classifier, program package, and runtime."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

PACKAGE_FORMAT = "abi-r20-instructional-realization-package/1"
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


class R20PackageError(RuntimeError):
    """Raised when an R20 package or prompt fails validation."""


def parse_prompt(prompt: str) -> tuple[str, dict[str, str]]:
    lines = prompt.splitlines()
    if len(lines) != 5 or not lines[0].startswith("INSTRUCTION: ") or lines[1] != "DATA:":
        raise R20PackageError("R20 prompt schema changed")
    instruction = lines[0].removeprefix("INSTRUCTION: ").strip()
    slots: dict[str, str] = {}
    for line in lines[2:]:
        key, separator, value = line.partition("=")
        if not separator or not key or not value or key in slots:
            raise R20PackageError("R20 data schema changed")
        slots[key] = value
    if set(slots) != {"field_a", "field_b", "field_c"}:
        raise R20PackageError("R20 slot inventory changed")
    return instruction, slots


def features(instruction: str) -> dict[str, int]:
    normalized = " ".join(re.findall(r"[a-z0-9]+", instruction.casefold()))
    padded = f"^^{normalized}$$"
    result: dict[str, int] = {}
    for width in (3, 4, 5):
        for index in range(len(padded) - width + 1):
            value = padded[index : index + width]
            result[value] = result.get(value, 0) + 1
    return result


def _cosine(left: dict[str, int], right: dict[str, int]) -> float:
    numerator = sum(value * right.get(key, 0) for key, value in left.items())
    left_norm = sum(value * value for value in left.values())
    right_norm = sum(value * value for value in right.values())
    if not left_norm or not right_norm:
        return 0.0
    return numerator / math.sqrt(left_norm * right_norm)


def load_package(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R20PackageError("R20 package is unreadable") from exc
    classes = value.get("classes")
    if (
        value.get("format") != PACKAGE_FORMAT
        or value.get("namespace") != "english/core/instructional-realization"
        or value.get("representation") != "instruction-ngram-centroid-plus-output-program"
        or not isinstance(classes, list)
        or len(classes) != 6
        or len({item.get("program") for item in classes if isinstance(item, dict)}) != 6
    ):
        raise R20PackageError("R20 package schema changed")
    for item in classes:
        program = item.get("program")
        centroid = item.get("centroid")
        if (
            not isinstance(program, str)
            or set(_PLACEHOLDER.findall(program)) != {"field_a", "field_b", "field_c"}
            or not isinstance(centroid, dict)
            or not centroid
            or any(
                not isinstance(key, str) or not isinstance(count, int) or count <= 0
                for key, count in centroid.items()
            )
            or not isinstance(item.get("support"), int)
            or item["support"] < 6
        ):
            raise R20PackageError("R20 package class changed")
    return value


def execute(package: dict[str, Any] | None, prompt: str) -> str | None:
    if package is None:
        return None
    instruction, slots = parse_prompt(prompt)
    observed = features(instruction)
    scored = [
        (_cosine(observed, item["centroid"]), index, item)
        for index, item in enumerate(package["classes"])
    ]
    _, _, selected = max(scored, key=lambda item: (item[0], -item[1]))
    return _PLACEHOLDER.sub(lambda match: slots[match.group(1)], selected["program"])
