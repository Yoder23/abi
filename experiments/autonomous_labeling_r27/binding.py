"""Fail-closed R27 held-out configuration bindings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, sha256_file


CODE_PATHS = (
    "experiments/autonomous_labeling_r27/PROTOCOL.md",
    "experiments/autonomous_labeling_r27/binding.py",
    "experiments/autonomous_labeling_r27/facts.py",
    "experiments/autonomous_labeling_r27/protocol.py",
    "experiments/autonomous_labeling_r27/public_qualification.py",
    "experiments/autonomous_labeling_r27/isolated_worker.py",
    "experiments/autonomous_labeling_r27/isolation.py",
    "experiments/autonomous_labeling_r27/isolation_smoke.py",
    "experiments/autonomous_labeling_r27/extraction_pivot_runner.sh",
    "experiments/autonomous_labeling_r27/source_run.py",
    "experiments/autonomous_labeling_r27/import_run.py",
    "experiments/autonomous_labeling_r27/verify.py",
    "experiments/autonomous_labeling_r27/prepare_secret.py",
    "experiments/autonomous_labeling_r27/prepare_import.py",
    "experiments/factual_semantic_r16/package.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/factual_semantic_r16/public_sequence_scoring.py",
    "experiments/foreign_capability_r14/core.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/generative_transfer_r21/live_verify_v7.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/semantic_replication_r23/binding.py",
    "experiments/semantic_replication_r23/protocol.py",
    "tests/test_autonomous_labeling_r27.py",
)

LAYERCAKE_PATHS = (
    "layercake/cake/installer.py", "layercake/cake/manifest.py", "layercake/cake/package.py",
    "layercake/cake/registry.py", "layercake/models/canonical_factual.py",
    "layercake/models/direct_cake_host.py", "layercake/models/portable_decoder.py",
)


def validate_file(root: Path, item: dict[str, Any], name: str) -> None:
    path = (root / str(item.get("path", ""))).resolve()
    if not path.is_file() or path.stat().st_size != item.get("bytes") or sha256_file(path) != item.get("sha256"):
        raise R14Error(f"R27 bound file changed: {name}")


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R27 config unreadable") from exc
    if (
        value.get("format") != "abi-r27-heldout-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("heldout_seed_commitment", "")))
        or value.get("facts_per_domain") != 3
        or value.get("extraction_views") != [0, 1, 2]
        or value.get("evaluation_views") != [0, 1, 2]
        or value.get("label_choices_supplied") != 0
        or value.get("training_authorized") is not False
    ):
        raise R14Error("R27 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R27 code inventory changed")
    for relative, digest in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != digest:
            raise R14Error(f"R27 code changed: {relative}")
    layercake_root = (root / "../layercake_release").resolve()
    for relative, digest in value.get("layercake_code_sha256", {}).items():
        target = layercake_root / relative
        if relative not in LAYERCAKE_PATHS or not target.is_file() or sha256_file(target) != digest:
            raise R14Error(f"R27 LayerCake code changed: {relative}")
    if set(value.get("layercake_code_sha256", {})) != set(LAYERCAKE_PATHS):
        raise R14Error("R27 LayerCake inventory changed")
    validate_file(root, value.get("public_prerequisite", {}), "public prerequisite")
    for name in ("r23_config", "r23_reveal", "r23_hidden_rows"):
        validate_file(root, value.get(name, {}), name)
    return value
