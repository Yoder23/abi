"""Fail-closed R24 configuration binding."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)

from .protocol import NAMESPACES, SEEDS, source_splits

CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/facts.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/generative_transfer_r21/live_verify_v7.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/semantic_replication_r23/protocol.py",
    "experiments/layercake_composition_r24/PROTOCOL.md",
    "experiments/layercake_composition_r24/protocol.py",
    "experiments/layercake_composition_r24/binding.py",
    "experiments/layercake_composition_r24/run.py",
    "experiments/layercake_composition_r24/verify.py",
    "experiments/layercake_composition_r24/freeze_config.py",
)

GATES = {
    "domain_exact_per_seed": 48,
    "namespace_route_exact": 48,
    "other_domain_target_answers": 0,
    "english_core_target_answers": 0,
    "english_rows_immutable_per_seed": 120,
    "cpu_gpu_exact_per_seed": 48,
    "packages_signed": 6,
    "package_lifecycle_exact": 6,
    "targeted_corruptions_rejected": 6,
}


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R24 config unreadable") from exc
    if (
        value.get("format") != "abi-r24-config/1"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))
        )
        or value.get("namespaces") != list(NAMESPACES)
        or value.get("seeds") != list(SEEDS)
        or value.get("training")
        != {
            "device": "cuda",
            "steps": 2000,
            "batch_size": 8,
            "learning_rate": 0.0008,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "row_exposure_per_package": 16000,
        }
        or value.get("model")
        != {
            "model_width": 48,
            "attention_heads": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "feedforward_width": 128,
            "pointer_width": 24,
            "dropout": 0.0,
            "maximum_source_lexemes": 128,
            "maximum_target_actions": 32,
        }
        or value.get("gates") != GATES
        or value.get("new_source_calls_authorized") is not False
        or value.get("english_package_changes_authorized") is not False
        or value.get("evaluation_training_authorized") is not False
    ):
        raise R14Error("R24 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R24 code inventory incomplete")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R24 code changed: {relative}")
    for name in (
        "r16_source_rows",
        "r16_receipt",
        "r16_strict_verification",
        "r23_config",
        "r23_reveal",
        "r23_engine",
        "r23_live_verification",
        "r23_hidden_rows",
    ):
        binding = value.get(name, {})
        target = root / str(binding.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != binding.get("bytes")
            or sha256_file(target) != binding.get("sha256")
        ):
            raise R14Error(f"R24 prerequisite changed: {name}")
    r16 = json_object(root / str(value["r16_strict_verification"]["path"]))
    r23 = json_object(root / str(value["r23_live_verification"]["path"]))
    if (
        r16.get("verdict") != "PASS"
        or r16.get("facts_exact") != 16
        or r16.get("evaluation_rows_exact") != 48
        or r23.get("status") != "PASS_STRICTLY_VERIFIED_SEMANTIC_REPLICATION"
        or r23.get("evidence_sha256") != selfless_evidence_hash(r23)
    ):
        raise R14Error("R24 prerequisite did not pass")
    source_splits(root / str(value["r16_source_rows"]["path"]))
    english = value.get("english_packages")
    if (
        not isinstance(english, list)
        or len(english) != 6
        or len({row.get("path") for row in english}) != 6
    ):
        raise R14Error("R24 English inventory changed")
    for package in english:
        target = root / str(package.get("path", ""))
        if (
            not target.is_file()
            or target.stat().st_size != package.get("bytes")
            or sha256_file(target) != package.get("sha256")
        ):
            raise R14Error("R24 English package changed")
    return value
