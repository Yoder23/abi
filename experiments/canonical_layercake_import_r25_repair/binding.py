"""Fail-closed binding for the R25 post-hoc scorer repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

CODE_PATHS = (
    "experiments/factual_semantic_r16/public_sequence_scoring.py",
    "experiments/canonical_layercake_import_r25_repair/PROTOCOL.md",
    "experiments/canonical_layercake_import_r25_repair/binding.py",
    "experiments/canonical_layercake_import_r25_repair/freeze_config.py",
    "experiments/canonical_layercake_import_r25_repair/verify.py",
)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R25 repair config unreadable") from exc
    if (
        value.get("format") != "abi-r25-semantic-repair-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("candidate_changes") != 0
        or value.get("gate_changes") != 0
        or value.get("scorer_changes") != 1
        or value.get("fresh_replication_required") is not True
    ):
        raise R14Error("R25 repair governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R25 repair code inventory changed")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R25 repair code changed: {relative}")
    for name in ("original_config", "result_inventory"):
        items = value.get(name)
        if not isinstance(items, list) or not items:
            raise R14Error(f"R25 repair {name} missing")
        if name == "original_config" and len(items) != 1:
            raise R14Error("R25 repair original config count changed")
        for item in items:
            target = root / str(item.get("path", ""))
            if (
                not target.is_file()
                or target.stat().st_size != item.get("bytes")
                or sha256_file(target) != item.get("sha256")
            ):
                raise R14Error(f"R25 repair evidence changed: {name}")
    return value
