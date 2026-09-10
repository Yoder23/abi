"""Canonical R18 factorized grammar package and generic realization runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.linguistic_realization_r17.frames import SIGNATURES, SLOT_KEYS

PACKAGE_FORMAT = "abi-r18-factorized-english-realization-package/1"
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


class R18PackageError(RuntimeError):
    """Raised when an R18 package or request violates its contract."""


def load_package(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R18PackageError("R18 package is unreadable") from exc
    templates = value.get("templates")
    support = value.get("support")
    if (
        value.get("format") != PACKAGE_FORMAT
        or value.get("namespace") != "english/core/realization"
        or value.get("factorized_axis") != "subject_number"
        or not isinstance(templates, dict)
        or set(templates) != set(SIGNATURES)
        or not isinstance(support, dict)
        or set(support) != set(SIGNATURES)
    ):
        raise R18PackageError("R18 package schema changed")
    for template in templates.values():
        if (
            not isinstance(template, str)
            or not template
            or not set(_PLACEHOLDER.findall(template)) <= set(SLOT_KEYS)
        ):
            raise R18PackageError("R18 template changed")
    return value


def realize(package: dict[str, Any] | None, signature: str, slots: dict[str, str]) -> str | None:
    if package is None:
        return None
    if signature not in SIGNATURES or set(slots) != set(SLOT_KEYS):
        raise R18PackageError("R18 realization request changed")
    template = str(package["templates"][signature])
    return _PLACEHOLDER.sub(lambda match: str(slots[match.group(1)]), template)
