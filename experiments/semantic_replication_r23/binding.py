"""Fail-closed configuration and seed custody for R23."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .protocol import seed_commitment, validate_public_lexicon

CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/instructional_realization_r20/protocol.py",
    "experiments/generative_transfer_r21/acquire.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/evaluate_hidden.py",
    "experiments/generative_transfer_r21/verify_hidden.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/semantic_replication_r23/PROTOCOL.md",
    "experiments/semantic_replication_r23/protocol.py",
    "experiments/semantic_replication_r23/binding.py",
    "experiments/semantic_replication_r23/acquire.py",
    "experiments/semantic_replication_r23/run.py",
    "experiments/semantic_replication_r23/verify.py",
    "experiments/semantic_replication_r23/freeze_config.py",
)

GATES = {
    "minimum_label_exact": 120,
    "minimum_task_functional_per_seed": 18,
    "minimum_non_hallucinating_per_seed": 120,
    "minimum_non_collapsed_per_seed": 120,
    "require_each_seed_no_worse_than_teacher": True,
    "require_each_seed_no_worse_than_controls": True,
    "bootstrap_samples": 10000,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R23 JSONL unreadable: {path}") from exc


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R23 config unreadable") from exc
    if (
        value.get("format") != "abi-r23-hidden-config/1"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))
        )
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("seed_commitment", "")))
        or value.get("rows") != {"total": 120, "per_task": 20, "tasks": 6}
        or value.get("methods")
        != ["raw_sequence", "labeled_monolith", "abi_factorized"]
        or value.get("seeds") != [21021, 21022, 21023]
        or value.get("gates") != GATES
        or value.get("student_retraining_authorized") is not False
        or value.get("package_changes_authorized") is not False
        or value.get("second_reveal_authorized") is not False
    ):
        raise R14Error("R23 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R23 code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R23 code changed: {relative}")
    for name in (
        "base_config",
        "public_candidate_engine",
        "public_candidate_labeler",
        "public_live_receipt",
        "public_strict_verification",
        "r21_hidden_result",
        "r21_hidden_verification",
        "r22_failed_receipt",
        "public_semantic_corpus",
    ):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R23 prerequisite changed: {name}")
    engine = json_object(root / str(value["public_candidate_engine"]["path"]))
    live = json_object(root / str(value["public_live_receipt"]["path"]))
    strict = json_object(root / str(value["public_strict_verification"]["path"]))
    hidden = json_object(root / str(value["r21_hidden_result"]["path"]))
    r22 = json_object(root / str(value["r22_failed_receipt"]["path"]))
    if (
        engine.get("verdict") != "PASS_PUBLIC_PREREQUISITE"
        or engine.get("evidence_sha256") != selfless_evidence_hash(engine)
        or live.get("status") != "PASS_FRESH_LIVE_PUBLIC_PREREQUISITE"
        or live.get("evidence_sha256") != selfless_evidence_hash(live)
        or strict.get("status") != "PASS_STRICTLY_VERIFIED_PUBLIC_PREREQUISITE"
        or strict.get("evidence_sha256") != selfless_evidence_hash(strict)
        or hidden.get("verdict") != "FAIL_HIDDEN_REPLICATION"
        or hidden.get("evidence_sha256") != selfless_evidence_hash(hidden)
        or r22.get("verdict") != "FAIL_NORMALIZED_SOURCE"
        or r22.get("evidence_sha256") != selfless_evidence_hash(r22)
    ):
        raise R14Error("R23 scientific prerequisite state changed")
    public_rows = _jsonl(root / str(value["public_semantic_corpus"]["path"]))
    if len(public_rows) != 600:
        raise R14Error("R23 public semantic corpus changed")
    try:
        counts = validate_public_lexicon(public_rows)
    except ValueError as exc:
        raise R14Error("R23 public semantic lexicon failed") from exc
    if value.get("public_marker_counts") != counts:
        raise R14Error("R23 semantic marker accounting changed")
    packages = value.get("packages")
    if (
        not isinstance(packages, list)
        or len(packages) != 24
        or len({row.get("path") for row in packages}) != 24
    ):
        raise R14Error("R23 package inventory changed")
    for package in packages:
        target = root / str(package.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != package.get("bytes")
            or sha256_file(target) != package.get("sha256")
        ):
            raise R14Error("R23 frozen package changed")
    return value


def load_seed_reveal(config: dict[str, Any], path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R23 seed reveal unreadable") from exc
    seed_hex = str(value.get("seed_hex", ""))
    if (
        value.get("format") != "abi-r23-hidden-seed-reveal/1"
        or seed_commitment(seed_hex) != config["seed_commitment"]
    ):
        raise R14Error("R23 seed reveal does not match commitment")
    return seed_hex
