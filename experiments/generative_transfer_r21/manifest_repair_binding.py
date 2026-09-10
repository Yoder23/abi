"""Fail-closed binding for the one R21 LayerCake manifest repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

MANIFEST_REPAIR_CODE_PATHS = (
    "experiments/generative_transfer_r21/MANIFEST_REPAIR_PROTOCOL.md",
    "experiments/generative_transfer_r21/manifest_repair_binding.py",
    "experiments/generative_transfer_r21/run_v5.py",
    "experiments/generative_transfer_r21/verify_v5.py",
    "experiments/generative_transfer_r21/freeze_manifest_repair.py",
)


def load_manifest_repair_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 manifest-repair config unreadable") from exc
    if (
        value.get("format") != "abi-r21-manifest-repair-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("input_contract_addition") != {"mode": "direct_selected_portable_decoder"}
        or value.get("retraining_authorized") is not True
        or value.get("data_changes_authorized") is not False
        or value.get("model_changes_authorized") is not False
        or value.get("gate_changes_authorized") is not False
        or value.get("layercake_changes_authorized") is not False
    ):
        raise R14Error("R21 manifest-repair governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(MANIFEST_REPAIR_CODE_PATHS):
        raise R14Error("R21 manifest-repair code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 manifest-repair code changed: {relative}")
    for name in ("hash_assurance_config", "failed_attempt_record"):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if not target.is_file() or sha256_file(target) != binding.get("sha256"):
            raise R14Error(f"R21 manifest-repair prerequisite changed: {name}")
    packages = value.get("failed_package_inventory")
    if not isinstance(packages, list) or len(packages) != 24:
        raise R14Error("R21 failed package inventory changed")
    for package in packages:
        target = root / str(package.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != package.get("bytes")
            or sha256_file(target) != package.get("sha256")
        ):
            raise R14Error("R21 failed package identity changed")
    return value
