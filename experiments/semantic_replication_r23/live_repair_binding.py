"""Fail-closed binding for the R23 live metadata-source repair."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .binding import load_config
from .live_binding import load_live_config

REPAIR_CODE_PATHS = (
    "experiments/semantic_replication_r23/LIVE_REPAIR_PROTOCOL.md",
    "experiments/semantic_replication_r23/live_repair_binding.py",
    "experiments/semantic_replication_r23/live_verify_v2.py",
    "experiments/semantic_replication_r23/verify_live_v2.py",
    "experiments/semantic_replication_r23/freeze_live_repair.py",
)


def load_repair_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R23 live-repair config unreadable") from exc
    if (
        value.get("format") != "abi-r23-live-repair-config/1"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))
        )
        or value.get("systems_source") != "frozen_public_candidate_engine"
        or value.get("candidate_changes_authorized") is not False
        or value.get("package_changes_authorized") is not False
        or value.get("scorer_changes_authorized") is not False
        or value.get("gate_changes_authorized") is not False
        or value.get("full_replay_required") is not True
    ):
        raise R14Error("R23 live-repair governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(REPAIR_CODE_PATHS):
        raise R14Error("R23 live-repair code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R23 live-repair code changed: {relative}")
    for name in ("live_config", "failed_attempt", "r23_config"):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R23 live-repair prerequisite changed: {name}")
    load_live_config(root, root / str(value["live_config"]["path"]))
    load_config(root, root / str(value["r23_config"]["path"]))
    failed = json_object(root / str(value["failed_attempt"]["path"]))
    if (
        failed.get("verdict") != "FAIL_CLOSED_BEFORE_EXECUTION"
        or failed.get("candidate_rows_executed") != 0
        or failed.get("exception_message") != "systems"
        or failed.get("evidence_sha256") != selfless_evidence_hash(failed)
    ):
        raise R14Error("R23 failed live attempt changed")
    return value
