"""Fail-closed binding and recomputation for R21 self-hash assurance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from experiments.foreign_capability_r14.core import R14Error, evidence_hash, sha256_file

ASSURANCE_CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/generative_transfer_r21/HASH_ASSURANCE_PROTOCOL.md",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/generative_transfer_r21/run_v4.py",
    "experiments/generative_transfer_r21/verify_v4.py",
    "experiments/generative_transfer_r21/freeze_hash_assurance.py",
)


def selfless_evidence_hash(value: Mapping[str, Any]) -> str:
    document = dict(value)
    document.pop("evidence_sha256", None)
    return evidence_hash(document)


def load_assurance_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 hash-assurance config unreadable") from exc
    if (
        value.get("format") != "abi-r21-hash-assurance-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("removed_fields") != ["evidence_sha256"]
        or value.get("data_changes_authorized") is not False
        or value.get("gate_changes_authorized") is not False
        or value.get("positive_stored-evidence_promotion_authorized") is not False
    ):
        raise R14Error("R21 hash-assurance governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(ASSURANCE_CODE_PATHS):
        raise R14Error("R21 hash-assurance code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 hash-assurance code changed: {relative}")
    for name in ("label_control_config", "source_receipt", "source_rows", "source_labeler"):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if not target.is_file() or sha256_file(target) != binding.get("sha256"):
            raise R14Error(f"R21 hash-assurance prerequisite changed: {name}")
    return value
