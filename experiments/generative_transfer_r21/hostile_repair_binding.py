"""Fail-closed binding for the R21 targeted hostile-control repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

HOSTILE_REPAIR_CODE_PATHS = (
    "experiments/generative_transfer_r21/HOSTILE_REPAIR_PROTOCOL.md",
    "experiments/generative_transfer_r21/hostile_repair_binding.py",
    "experiments/generative_transfer_r21/live_verify_v7.py",
    "experiments/generative_transfer_r21/verify_live_v7.py",
    "experiments/generative_transfer_r21/freeze_hostile_repair.py",
)


def load_hostile_repair_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 hostile-repair config unreadable") from exc
    if (
        value.get("format") != "abi-r21-hostile-repair-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("old_mutation") != "final_tar_padding_byte"
        or value.get("new_mutation") != "middle_byte_of_tensors.safetensors"
        or value.get("candidate_changes_authorized") is not False
        or value.get("gate_changes_authorized") is not False
        or value.get("full_replay_required") is not True
    ):
        raise R14Error("R21 hostile-repair governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(HOSTILE_REPAIR_CODE_PATHS):
        raise R14Error("R21 hostile-repair code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 hostile-repair code changed: {relative}")
    for name in ("live_verification_config", "failed_verifier_record"):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R21 hostile-repair prerequisite changed: {name}")
    return value
