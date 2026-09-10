"""Fail-closed binding for R21 fresh live verification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

LIVE_CODE_PATHS = (
    "experiments/generative_transfer_r21/LIVE_VERIFICATION_PROTOCOL.md",
    "experiments/generative_transfer_r21/live_binding.py",
    "experiments/generative_transfer_r21/live_verify_v6.py",
    "experiments/generative_transfer_r21/freeze_live_verification.py",
)


def _validate_inventory(root: Path, rows: Any, expected: int, label: str) -> None:
    if not isinstance(rows, list) or len(rows) != expected:
        raise R14Error(f"R21 {label} inventory count changed")
    if len({str(row.get("path", "")) for row in rows}) != expected:
        raise R14Error(f"R21 {label} inventory is duplicated")
    for row in rows:
        target = root / str(row.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != row.get("bytes")
            or sha256_file(target) != row.get("sha256")
        ):
            raise R14Error(f"R21 {label} inventory changed")


def load_live_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 live-verification config unreadable") from exc
    required = {
        "gpu_rows": 1080,
        "cpu_rows": 54,
        "package_removals": 24,
        "package_restorations": 24,
        "corrupt_packages_rejected": 24,
        "signed_packages": 24,
        "source_software_loaded": False,
    }
    if (
        value.get("format") != "abi-r21-live-verification-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or value.get("candidate_verdict") != "PASS_PUBLIC_PREREQUISITE"
        or value.get("required") != required
        or value.get("stored_scientific_booleans_trusted") is not False
    ):
        raise R14Error("R21 live-verification governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(LIVE_CODE_PATHS):
        raise R14Error("R21 live-verification code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 live-verification code changed: {relative}")
    for name in (
        "manifest_repair_config",
        "candidate_wrapper",
        "candidate_engine",
        "candidate_observations",
        "candidate_labeler",
        "assurance_binding",
        "manifest_repair_binding",
    ):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R21 live-verification prerequisite changed: {name}")
    _validate_inventory(root, value.get("packages"), 24, "package")
    layercake = value.get("layercake_python")
    if (
        not isinstance(layercake, list)
        or value.get("layercake_python_file_count") != len(layercake)
        or len(layercake) < 10
    ):
        raise R14Error("R21 LayerCake code inventory is absent")
    _validate_inventory(root, layercake, len(layercake), "LayerCake code")
    return value
