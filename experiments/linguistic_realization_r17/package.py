"""Canonical R17 template package and generic teacher-free realizer."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .frames import SIGNATURES, SLOT_KEYS

PACKAGE_FORMAT = "abi-r17-canonical-english-realization-package/1"
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


class R17PackageError(RuntimeError):
    """Raised when an R17 package or request violates its contract."""


def load_package(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R17PackageError("R17 package is unreadable") from exc
    templates = value.get("templates")
    if (
        value.get("format") != PACKAGE_FORMAT
        or value.get("namespace") != "english/core/realization"
        or not isinstance(templates, dict)
        or set(templates) != set(SIGNATURES)
        or any(not isinstance(item, str) or not item for item in templates.values())
    ):
        raise R17PackageError("R17 package schema changed")
    for template in templates.values():
        if not set(_PLACEHOLDER.findall(template)) <= set(SLOT_KEYS):
            raise R17PackageError("R17 package contains an unknown slot")
    return value


def realize(package: dict[str, Any] | None, signature: str, slots: dict[str, str]) -> str | None:
    if package is None:
        return None
    if signature not in SIGNATURES or set(slots) != set(SLOT_KEYS):
        raise R17PackageError("R17 realization request changed")
    if any(not isinstance(value, str) or not value for value in slots.values()):
        raise R17PackageError("R17 slot values must be non-empty strings")
    template = str(package["templates"][signature])
    return _PLACEHOLDER.sub(lambda match: slots[match.group(1)], template)
