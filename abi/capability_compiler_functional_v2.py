"""Prospective surface-equivalence normalization for functional evaluators."""

from __future__ import annotations

import copy
import re
from typing import Any

from .capability_compiler_phase2_common import evaluate_functional


NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalize_surface(value: str) -> str:
    value = value.casefold()
    for word, digit in NUMBER_WORDS.items():
        value = re.sub(rf"\b{word}\b", digit, value)
    return value


def normalize_evaluator(evaluator: dict[str, Any], capability: str) -> dict[str, Any]:
    result = copy.deepcopy(evaluator)
    kind = result["kind"]
    if kind in {"contains_all", "contains_any", "ordered_contains"}:
        result["values"] = [normalize_surface(str(value)) for value in result["values"]]
    elif kind == "exact":
        result["value"] = normalize_surface(str(result["value"]))
    elif kind == "all_of":
        result["rules"] = [normalize_evaluator(rule, capability) for rule in result["rules"]]
    if capability == "abstention" and kind == "contains_any":
        result["values"] = list(result["values"]) + ["cannot be known", "can not be known"]
    return result


def evaluate_functional_v2(output: str, evaluator: dict[str, Any], capability: str) -> bool:
    return evaluate_functional(normalize_surface(output), normalize_evaluator(evaluator, capability))
