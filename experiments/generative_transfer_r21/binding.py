"""Fail-closed R21 configuration binding."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/instructional_realization_r20/protocol.py",
    "experiments/generative_transfer_r21/PROTOCOL.md",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/binding.py",
    "experiments/generative_transfer_r21/acquire.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/verify.py",
    "experiments/generative_transfer_r21/freeze_config.py",
)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 configuration unreadable") from exc
    if (
        value.get("format") != "abi-r21-public-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("source", {}).get("model_id") != "Qwen/Qwen2-7B-Instruct"
        or value.get("data") != {"training_rows": 600, "evaluation_rows": 120, "tasks": 6}
        or value.get("methods") != ["raw_sequence", "labeled_monolith", "abi_factorized"]
        or value.get("seeds") != [21021, 21022, 21023]
    ):
        raise R14Error("R21 configuration changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R21 code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 bound code changed: {relative}")
    return value
