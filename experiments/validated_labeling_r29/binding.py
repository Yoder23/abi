"""Fail-closed R29 held-out configuration."""

from __future__ import annotations

import json
import re
from pathlib import Path

from experiments.foreign_capability_r14.core import R14Error, sha256_file


CODE_PATHS = (
    "experiments/validated_labeling_r29/PROTOCOL.md",
    "experiments/validated_labeling_r29/binding.py",
    "experiments/validated_labeling_r29/facts.py",
    "experiments/validated_labeling_r29/protocol.py",
    "experiments/validated_labeling_r29/isolated_worker.py",
    "experiments/validated_labeling_r29/isolation.py",
    "experiments/validated_labeling_r29/extraction_pivot_runner.sh",
    "experiments/validated_labeling_r29/public_qualification.py",
    "experiments/validated_labeling_r29/source_run.py",
    "experiments/validated_labeling_r29/import_run.py",
    "experiments/validated_labeling_r29/prepare_secret.py",
    "experiments/validated_labeling_r29/verify.py",
    "tests/test_validated_labeling_r29.py",
    "experiments/autonomous_labeling_r27/facts.py",
    "experiments/autonomous_labeling_r27/binding.py",
    "experiments/autonomous_labeling_r27/import_run.py",
    "experiments/autonomous_labeling_r27/prepare_import.py",
    "experiments/autonomous_labeling_r27/public_qualification.py",
    "experiments/autonomous_labeling_r27/source_run.py",
    "experiments/robust_labeling_r28/protocol.py",
    "experiments/factual_semantic_r16/package.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/foreign_capability_r14/core.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/generative_transfer_r21/live_verify_v7.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/semantic_replication_r23/binding.py",
    "experiments/semantic_replication_r23/protocol.py",
)


def validate_file(root, item, name):
    path = (root / str(item.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"):
        raise R14Error(f"R29 bound file changed: {name}")


def load_config(root: Path, path: Path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R29 config unreadable") from exc
    if (
        value.get("format") != "abi-r29-heldout-config/3"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("heldout_seed_commitment", "")))
        or value.get("facts_per_domain") != 3
        or not isinstance(value.get("excluded_fact_ids"), list)
        or len(value["excluded_fact_ids"]) != 24
        or len(set(value["excluded_fact_ids"])) != 24
        or value.get("label_choices_supplied") != 0
        or value.get("validator") != "safe-stdlib-ast/1"
        or value.get("minimum_parse") != 68
        or value.get("minimum_unvalidated_raw_exact") != 32
        or value.get("training_authorized") is not False
    ):
        raise R14Error("R29 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R29 code inventory changed")
    for relative, digest in code.items():
        if not (root / relative).is_file() or sha256_file(root / relative) != digest:
            raise R14Error(f"R29 code changed: {relative}")
    for name in ("public_prerequisite", "r27_host_config"):
        validate_file(root, value.get(name, {}), name)
    return value
