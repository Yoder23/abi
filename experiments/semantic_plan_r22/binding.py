"""Fail-closed configuration binding for R22."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/verify.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/semantic_plan_r22/PROTOCOL.md",
    "experiments/semantic_plan_r22/protocol.py",
    "experiments/semantic_plan_r22/binding.py",
    "experiments/semantic_plan_r22/normalize.py",
    "experiments/semantic_plan_r22/run.py",
    "experiments/semantic_plan_r22/verify.py",
    "experiments/semantic_plan_r22/freeze_config.py",
)


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R22 config unreadable") from exc
    gates = {
        "minimum_functional": 594,
        "minimum_non_hallucinating": 600,
        "minimum_non_collapsed": 594,
        "required_fields_verbatim_once": 600,
        "required_label_exact": 600,
    }
    if (
        value.get("format") != "abi-r22-config/1"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))
        )
        or value.get("rows") != 600
        or value.get("normalization_calls") != 600
        or value.get("gates") != gates
        or value.get("retry_authorized") is not False
        or value.get("evaluation_access_authorized") is not False
        or value.get("student_training_before_source_pass") is not False
    ):
        raise R14Error("R22 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R22 code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R22 code changed: {relative}")
    for name in (
        "base_training_config",
        "raw_source_receipt",
        "raw_source_rows",
        "r21_hidden_result",
        "r21_hidden_verification",
    ):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R22 prerequisite changed: {name}")
    return value
