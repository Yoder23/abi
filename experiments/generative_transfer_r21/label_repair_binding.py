"""Fail-closed binding for the single R21 label-interface repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

REPAIR_CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/generative_transfer_r21/LABEL_REPAIR_PROTOCOL.md",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/binding.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/verify.py",
    "experiments/generative_transfer_r21/label_repair_binding.py",
    "experiments/generative_transfer_r21/acquire_labels_v2.py",
    "experiments/generative_transfer_r21/run_v2.py",
    "experiments/generative_transfer_r21/verify_v2.py",
    "experiments/generative_transfer_r21/freeze_label_repair.py",
)


def load_repair_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 label-repair config unreadable") from exc
    if (
        value.get("format") != "abi-r21-label-repair-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("candidate_letters")
        != {
            "A": "prose",
            "B": "summary",
            "C": "email",
            "D": "bullets",
            "E": "clarification",
            "F": "abstention",
        }
        or value.get("minimum_exact_labels") != 594
        or value.get("response_regeneration_authorized") is not False
        or value.get("evaluation_access_authorized") is not False
    ):
        raise R14Error("R21 label-repair governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(REPAIR_CODE_PATHS):
        raise R14Error("R21 label-repair code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 label-repair code changed: {relative}")
    for name in ("base_config", "failed_source_receipt", "failed_source_rows"):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if not target.is_file() or sha256_file(target) != binding.get("sha256"):
            raise R14Error(f"R21 label-repair prerequisite changed: {name}")
    return value
