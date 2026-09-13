"""Fail-closed R28 configuration binding."""

from __future__ import annotations

import json, re
from pathlib import Path

from experiments.foreign_capability_r14.core import R14Error, sha256_file


CODE_PATHS = (
    "experiments/robust_labeling_r28/PROTOCOL.md", "experiments/robust_labeling_r28/binding.py",
    "experiments/robust_labeling_r28/facts.py", "experiments/robust_labeling_r28/protocol.py",
    "experiments/robust_labeling_r28/isolated_worker.py", "experiments/robust_labeling_r28/isolation.py",
    "experiments/robust_labeling_r28/extraction_pivot_runner.sh", "experiments/robust_labeling_r28/public_qualification.py",
    "experiments/robust_labeling_r28/source_run.py", "experiments/robust_labeling_r28/prepare_secret.py",
    "experiments/robust_labeling_r28/verify.py", "tests/test_robust_labeling_r28.py",
    "experiments/autonomous_labeling_r27/facts.py", "experiments/autonomous_labeling_r27/protocol.py",
    "experiments/autonomous_labeling_r27/public_qualification.py", "experiments/autonomous_labeling_r27/source_run.py",
    "experiments/factual_semantic_r16/package.py", "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/foreign_capability_r14/core.py", "experiments/generative_transfer_r21/hash_assurance_binding.py",
)


def validate_file(root, item, name):
    path = (root / str(item.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"): raise R14Error(f"R28 bound file changed: {name}")


def load_config(root: Path, path: Path):
    try: value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc: raise R14Error("R28 config unreadable") from exc
    if value.get("format") != "abi-r28-heldout-config/1" or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", ""))) or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("heldout_seed_commitment", ""))) or value.get("facts_per_domain") != 3 or value.get("label_choices_supplied") != 0 or value.get("quorum") != 2 or value.get("minimum_raw_exact") != 68 or value.get("minimum_teacher_agreement") != 32 or value.get("training_authorized") is not False: raise R14Error("R28 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS): raise R14Error("R28 code inventory changed")
    for relative, digest in code.items():
        if not (root / relative).is_file() or sha256_file(root / relative) != digest: raise R14Error(f"R28 code changed: {relative}")
    for name in ("public_prerequisite", "r27_host_config"):
        validate_file(root, value.get(name, {}), name)
    return value

