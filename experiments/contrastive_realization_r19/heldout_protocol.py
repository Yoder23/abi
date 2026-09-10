"""Fail-closed code, evidence, and secret bindings for R19 holdout."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file

CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/linguistic_realization_r17/frames.py",
    "experiments/linguistic_realization_r17/frames_v2.py",
    "experiments/linguistic_realization_r17/public_qualification.py",
    "experiments/linguistic_realization_r17/verify_source.py",
    "experiments/functional_realization_r18/run_public.py",
    "experiments/functional_realization_r18/verify.py",
    "experiments/contrastive_realization_r19/HOLDOUT_PROTOCOL.md",
    "experiments/contrastive_realization_r19/PUBLIC_PROTOCOL_V4.md",
    "experiments/contrastive_realization_r19/package.py",
    "experiments/contrastive_realization_r19/compiler.py",
    "experiments/contrastive_realization_r19/isolated_worker.py",
    "experiments/contrastive_realization_r19/extraction_pivot_runner.sh",
    "experiments/contrastive_realization_r19/isolation.py",
    "experiments/contrastive_realization_r19/run_development.py",
    "experiments/contrastive_realization_r19/verify_development.py",
    "experiments/contrastive_realization_r19/hidden_frames.py",
    "experiments/contrastive_realization_r19/heldout_protocol.py",
    "experiments/contrastive_realization_r19/prepare_secret.py",
    "experiments/contrastive_realization_r19/freeze_config.py",
    "experiments/contrastive_realization_r19/acquire_heldout.py",
    "experiments/contrastive_realization_r19/verify_heldout_source.py",
    "experiments/contrastive_realization_r19/run_heldout.py",
    "experiments/contrastive_realization_r19/verify_heldout.py",
    "experiments/contrastive_realization_r19/hostile_audit_heldout.py",
    "experiments/contrastive_realization_r19/verify_heldout_live.py",
)

DEVELOPMENT_EVIDENCE = {
    "receipt": "results/contrastive_realization_r19/development_v5_complete/receipt.json",
    "strict": (
        "results/contrastive_realization_r19/development_v5_complete/strict_verification.json"
    ),
    "hostile": ("results/contrastive_realization_r19/development_v5_complete/hostile_audit.json"),
    "live": (
        "results/contrastive_realization_r19/development_v5_live_replay/live_verification.json"
    ),
}

SOURCE = {
    "model_id": "Qwen/Qwen2-7B-Instruct",
    "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
    "max_new_tokens": 48,
    "device": "cuda",
    "snapshot_evidence_sha256": (
        "bdb164548ca29e580e465f520df01fce77524eeec01e807c413443751b4f004b"
    ),
}

DATA = {
    "signatures": 24,
    "extraction_rows_per_signature": 3,
    "evaluation_rows_per_signature": 2,
    "extraction_rows": 72,
    "evaluation_rows": 48,
    "source_rows": 120,
    "lexeme_bank": 20,
    "lexically_disjoint_from_r17_r18": True,
}

GATES = {
    "minimum_parseable_per_signature": 2,
    "package_functional_exact": 48,
    "removed_abstain": 48,
    "max_control_accuracy": 0.10,
    "max_source_exact_regressions": 0,
    "require_package_at_least_modal": True,
}

PHYSICAL_EXTRACTION = {
    "distribution": "Ubuntu",
    "policy": "linux-pivot-root-no-network/1",
}


def load_bound_inputs(
    root: Path, config_path: Path, reveal_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    if (
        config.get("format") != "abi-r19-heldout-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(config.get("implementation_freeze_commit")))
        or config.get("source") != SOURCE
        or config.get("data") != DATA
        or config.get("gates") != GATES
        or config.get("physical_extraction") != PHYSICAL_EXTRACTION
    ):
        raise R14Error("R19 held-out config changed")
    code = config.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R19 held-out code inventory missing")
    for relative, expected in code.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise R14Error(f"R19 held-out code changed: {relative}")
    if (
        not reveal_path.is_file()
        or sha256_file(reveal_path) != config.get("reveal_sha256")
        or reveal.get("format") != "abi-r19-heldout-reveal/1"
        or reveal.get("commitment") != config.get("heldout_seed_commitment")
    ):
        raise R14Error("R19 held-out reveal changed")
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except (KeyError, ValueError) as exc:
        raise R14Error("R19 held-out secret changed") from exc
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != reveal["commitment"]:
        raise R14Error("R19 held-out commitment mismatch")
    prerequisites = config.get("development_prerequisite")
    if not isinstance(prerequisites, dict) or set(prerequisites) != set(DEVELOPMENT_EVIDENCE):
        raise R14Error("R19 development prerequisite inventory missing")
    for name, expected_relative in DEVELOPMENT_EVIDENCE.items():
        item = prerequisites[name]
        if not isinstance(item, dict) or item.get("path") != expected_relative:
            raise R14Error(f"R19 development prerequisite changed: {name}")
        path = root / expected_relative
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise R14Error(f"R19 development prerequisite changed: {name}")
    return config, reveal
