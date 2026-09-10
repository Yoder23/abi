"""Canonical R16 factual package and generic teacher-free executor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import sha256_file
from experiments.preexisting_representation_r15b.public_qualification import canonical_json_bytes

from .facts import namespace_from_question


class R16PackageError(RuntimeError):
    """Raised when a factual package or query violates the contract."""


def write_package_once(path: Path, namespace: str, facts: list[dict[str, str]]) -> dict[str, Any]:
    normalized = sorted(facts, key=lambda item: (item["relation"], item["entity"].casefold()))
    if len({(item["relation"], item["entity"].casefold()) for item in normalized}) != len(normalized):
        raise R16PackageError("duplicate R16 fact key")
    payload = {
        "format": "abi-r16-canonical-factual-package/1",
        "namespace": namespace,
        "facts": normalized,
    }
    if path.exists():
        raise R16PackageError(f"immutable R16 package exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_package(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R16PackageError("R16 package is unreadable") from exc
    if payload.get("format") != "abi-r16-canonical-factual-package/1":
        raise R16PackageError("R16 package format changed")
    if not isinstance(payload.get("namespace"), str) or not isinstance(payload.get("facts"), list):
        raise R16PackageError("R16 package schema changed")
    return payload


def answer(packages: list[dict[str, Any]], query: str) -> str | None:
    namespace = namespace_from_question(query)
    candidates = []
    for package in packages:
        if package["namespace"] != namespace:
            continue
        for fact in package["facts"]:
            if str(fact["entity"]).casefold() in query.casefold():
                candidates.append(str(fact["value"]))
    if len(candidates) > 1:
        raise R16PackageError("R16 query matches multiple facts")
    return candidates[0] if candidates else None
