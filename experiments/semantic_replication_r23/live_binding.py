"""Fail-closed binding for R23 fresh live verification."""

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

LIVE_CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/live_verify_v7.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/semantic_replication_r23/protocol.py",
    "experiments/semantic_replication_r23/binding.py",
    "experiments/semantic_replication_r23/acquire.py",
    "experiments/semantic_replication_r23/verify.py",
    "experiments/semantic_replication_r23/LIVE_PROTOCOL.md",
    "experiments/semantic_replication_r23/live_binding.py",
    "experiments/semantic_replication_r23/live_verify.py",
    "experiments/semantic_replication_r23/verify_live.py",
    "experiments/semantic_replication_r23/freeze_live.py",
)


def load_live_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R23 live config unreadable") from exc
    if (
        value.get("format") != "abi-r23-live-config/1"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))
        )
        or value.get("full_gpu_replay_required") is not True
        or value.get("cpu_task_cover_required") is not True
        or value.get("lifecycle_replay_required") is not True
        or value.get("targeted_corruption") != "middle_byte_of_tensors.safetensors"
        or value.get("candidate_changes_authorized") is not False
        or value.get("gate_changes_authorized") is not False
    ):
        raise R14Error("R23 live governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(LIVE_CODE_PATHS):
        raise R14Error("R23 live code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R23 live code changed: {relative}")
    for name in (
        "r23_config",
        "seed_reveal",
        "source_receipt",
        "source_rows",
        "candidate_wrapper",
        "candidate_engine",
        "candidate_observations",
        "candidate_cpu_observations",
        "stored_strict_verification",
    ):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R23 live prerequisite changed: {name}")
    config = load_config(root, root / str(value["r23_config"]["path"]))
    strict = json_object(root / str(value["stored_strict_verification"]["path"]))
    if (
        strict.get("status") != "PASS_VERIFIED_SEMANTIC_REPLICATION"
        or strict.get("evidence_sha256") != selfless_evidence_hash(strict)
        or len(config["packages"]) != 24
    ):
        raise R14Error("R23 live positive prerequisite changed")
    return value
