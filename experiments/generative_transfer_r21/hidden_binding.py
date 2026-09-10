"""Fail-closed config and seed custody for R21 hidden replication."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file

from .hash_assurance_binding import selfless_evidence_hash
from .hidden_protocol import seed_commitment

HIDDEN_CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/instructional_realization_r20/protocol.py",
    "experiments/generative_transfer_r21/HIDDEN_PROTOCOL.md",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/acquire.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/generative_transfer_r21/hidden_protocol.py",
    "experiments/generative_transfer_r21/hidden_binding.py",
    "experiments/generative_transfer_r21/acquire_hidden.py",
    "experiments/generative_transfer_r21/evaluate_hidden.py",
    "experiments/generative_transfer_r21/verify_hidden.py",
    "experiments/generative_transfer_r21/freeze_hidden.py",
)


def load_hidden_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 hidden config unreadable") from exc
    gates = {
        "minimum_label_exact": 114,
        "minimum_task_functional_per_seed": 15,
        "minimum_non_hallucinating_per_seed": 114,
        "minimum_non_collapsed_per_seed": 114,
        "require_each_seed_no_worse_than_teacher": True,
        "require_each_seed_no_worse_than_controls": True,
        "bootstrap_samples": 10000,
    }
    if (
        value.get("format") != "abi-r21-hidden-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("seed_commitment", "")))
        or value.get("rows") != {"total": 120, "per_task": 20, "tasks": 6}
        or value.get("methods") != ["raw_sequence", "labeled_monolith", "abi_factorized"]
        or value.get("seeds") != [21021, 21022, 21023]
        or value.get("gates") != gates
        or value.get("student_retraining_authorized") is not False
        or value.get("second_reveal_authorized") is not False
    ):
        raise R14Error("R21 hidden governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(HIDDEN_CODE_PATHS):
        raise R14Error("R21 hidden code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R21 hidden code changed: {relative}")
    for name in (
        "base_config",
        "public_candidate_wrapper",
        "public_candidate_engine",
        "public_candidate_labeler",
        "public_live_receipt",
        "public_strict_verification",
    ):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R21 hidden prerequisite changed: {name}")
    try:
        engine = json.loads(
            (root / str(value["public_candidate_engine"]["path"])).read_text(encoding="utf-8")
        )
        live = json.loads(
            (root / str(value["public_live_receipt"]["path"])).read_text(encoding="utf-8")
        )
        strict = json.loads(
            (root / str(value["public_strict_verification"]["path"])).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 hidden public prerequisite unreadable") from exc
    if (
        engine.get("verdict") != "PASS_PUBLIC_PREREQUISITE"
        or engine.get("evidence_sha256") != selfless_evidence_hash(engine)
        or live.get("status") != "PASS_FRESH_LIVE_PUBLIC_PREREQUISITE"
        or live.get("evidence_sha256") != selfless_evidence_hash(live)
        or strict.get("status") != "PASS_STRICTLY_VERIFIED_PUBLIC_PREREQUISITE"
        or strict.get("evidence_sha256") != selfless_evidence_hash(strict)
    ):
        raise R14Error("R21 hidden public prerequisite did not pass")
    packages = value.get("packages")
    if (
        not isinstance(packages, list)
        or len(packages) != 24
        or len({row.get("path") for row in packages}) != 24
    ):
        raise R14Error("R21 hidden package inventory changed")
    for package in packages:
        target = root / str(package.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != package.get("bytes")
            or sha256_file(target) != package.get("sha256")
        ):
            raise R14Error("R21 hidden package identity changed")
    return value


def load_seed_reveal(config: dict[str, Any], path: Path) -> str:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 hidden seed reveal unreadable") from exc
    seed_hex = str(value.get("seed_hex", ""))
    if (
        value.get("format") != "abi-r21-hidden-seed-reveal/1"
        or seed_commitment(seed_hex) != config["seed_commitment"]
    ):
        raise R14Error("R21 hidden seed reveal does not match commitment")
    return seed_hex
