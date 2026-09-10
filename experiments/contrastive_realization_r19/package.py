"""R19 contrastive package schema and generic slot runtime."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.linguistic_realization_r17.frames import SIGNATURES, SLOT_KEYS

PACKAGE_FORMAT = "abi-r19-contrastive-english-realization-package/1"
_PLACEHOLDER = re.compile(r"\{([a-z0-9_]+)\}")


class R19PackageError(RuntimeError):
    """Raised when an R19 package or realization request is invalid."""


def load_package(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R19PackageError("R19 package is unreadable") from exc
    if (
        value.get("format") != PACKAGE_FORMAT
        or value.get("namespace") != "english/core/realization"
        or value.get("factorization") != "same-number-polarity-contrast"
        or not isinstance(value.get("templates"), dict)
        or set(value["templates"]) != set(SIGNATURES)
        or not isinstance(value.get("support"), dict)
        or set(value["support"]) != set(SIGNATURES)
    ):
        raise R19PackageError("R19 package schema changed")
    for template in value["templates"].values():
        if (
            not isinstance(template, str)
            or not template
            or not set(_PLACEHOLDER.findall(template)) <= set(SLOT_KEYS)
        ):
            raise R19PackageError("R19 template changed")
    return value


def realize(package: dict[str, Any] | None, signature: str, slots: dict[str, str]) -> str | None:
    if package is None:
        return None
    if signature not in SIGNATURES or set(slots) != set(SLOT_KEYS):
        raise R19PackageError("R19 realization request changed")
    template = str(package["templates"][signature])
    return _PLACEHOLDER.sub(lambda match: str(slots[match.group(1)]), template)
