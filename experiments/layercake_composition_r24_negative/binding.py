"""Fail-closed binding for additive R24 negative-result verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

CODE_PATHS = (
    "experiments/layercake_composition_r24_negative/PROTOCOL.md",
    "experiments/layercake_composition_r24_negative/binding.py",
    "experiments/layercake_composition_r24_negative/freeze_config.py",
    "experiments/layercake_composition_r24_negative/verify.py",
)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R24 negative-verification config unreadable") from exc
    if (
        value.get("format") != "abi-r24-negative-verification-config/1"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))
        )
        or value.get("candidate_changes") != 0
        or value.get("gate_changes") != 0
        or value.get("scorer_changes") != 0
    ):
        raise R14Error("R24 negative-verification governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R24 negative-verification code inventory changed")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R24 negative-verification code changed: {relative}")
    for name in ("original_config", "result_inventory"):
        binding = value.get(name)
        if name == "original_config":
            bindings = [binding]
        else:
            bindings = binding
        if not isinstance(bindings, list) or not bindings:
            raise R14Error(f"R24 negative-verification {name} inventory missing")
        for item in bindings:
            if not isinstance(item, dict):
                raise R14Error(f"R24 negative-verification {name} binding malformed")
            target = root / str(item.get("path", ""))
            if (
                not target.is_file()
                or target.stat().st_size != item.get("bytes")
                or sha256_file(target) != item.get("sha256")
            ):
                raise R14Error(f"R24 negative-verification evidence changed: {name}")
    return value
