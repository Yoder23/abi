"""Fail-closed public implementation and protocol binding for R20."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file

CODE_PATHS = (
    "experiments/foreign_capability_r14/core.py",
    "experiments/factual_semantic_r16/public_qualification.py",
    "experiments/linguistic_realization_r17/verify_source.py",
    "experiments/instructional_realization_r20/PUBLIC_PROTOCOL.md",
    "experiments/instructional_realization_r20/protocol.py",
    "experiments/instructional_realization_r20/package.py",
    "experiments/instructional_realization_r20/compiler.py",
    "experiments/instructional_realization_r20/isolated_worker.py",
    "experiments/instructional_realization_r20/extraction_pivot_runner.sh",
    "experiments/instructional_realization_r20/isolation.py",
    "experiments/instructional_realization_r20/binding.py",
    "experiments/instructional_realization_r20/freeze_config.py",
    "experiments/instructional_realization_r20/source_acquisition.py",
    "experiments/instructional_realization_r20/verify_source.py",
    "experiments/instructional_realization_r20/run_public.py",
    "experiments/instructional_realization_r20/verify.py",
    "experiments/instructional_realization_r20/hostile_audit.py",
    "experiments/instructional_realization_r20/verify_live.py",
)

SOURCE = {
    "model_id": "Qwen/Qwen2-7B-Instruct",
    "revision": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
    "max_new_tokens": 96,
    "device": "cuda",
    "snapshot_evidence_sha256": (
        "bdb164548ca29e580e465f520df01fce77524eeec01e807c413443751b4f004b"
    ),
}

DATA = {
    "tasks": 6,
    "extraction_rows": 72,
    "evaluation_rows": 120,
    "evaluation_rows_per_task": 20,
    "distinct_prompts": 192,
}

GATES = {
    "package_functional_exact": 120,
    "minimum_package_exact_per_task": 19,
    "max_source_exact_regressions": 0,
    "require_package_at_least_source": True,
    "max_control_functional_exact": 24,
    "removed_abstain": 120,
}

PHYSICAL_EXTRACTION = {
    "distribution": "Ubuntu",
    "policy": "linux-pivot-root-no-network/1",
}


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = json_object(config_path)
    if (
        config.get("format") != "abi-r20-public-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(config.get("implementation_freeze_commit")))
        or config.get("source") != SOURCE
        or config.get("data") != DATA
        or config.get("gates") != GATES
        or config.get("physical_extraction") != PHYSICAL_EXTRACTION
    ):
        raise R14Error("R20 public config changed")
    code = config.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R20 public code inventory changed")
    for relative, expected in code.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise R14Error(f"R20 public code changed: {relative}")
    return config
