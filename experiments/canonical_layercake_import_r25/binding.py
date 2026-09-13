"""Fail-closed R25 configuration binding."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

from .protocol import BUILD_INITIALIZATIONS, HOST_INITIALIZATIONS, NAMESPACES, source_splits

CODE_PATHS = (
    "experiments/factual_semantic_r16/package.py",
    "experiments/foreign_capability_r14/core.py",
    "experiments/generative_transfer_r21/hash_assurance_binding.py",
    "experiments/generative_transfer_r21/live_verify_v7.py",
    "experiments/generative_transfer_r21/protocol.py",
    "experiments/generative_transfer_r21/run.py",
    "experiments/layercake_composition_r24/protocol.py",
    "experiments/semantic_replication_r23/protocol.py",
    "experiments/canonical_layercake_import_r25/PROTOCOL.md",
    "experiments/canonical_layercake_import_r25/protocol.py",
    "experiments/canonical_layercake_import_r25/binding.py",
    "experiments/canonical_layercake_import_r25/freeze_config.py",
    "experiments/canonical_layercake_import_r25/run.py",
    "experiments/canonical_layercake_import_r25/verify.py",
)

LAYERCAKE_PATHS = (
    "layercake/cake/installer.py",
    "layercake/cake/manifest.py",
    "layercake/cake/package.py",
    "layercake/cake/registry.py",
    "layercake/models/canonical_factual.py",
    "layercake/models/direct_cake_host.py",
    "layercake/models/portable_decoder.py",
)

GATES = {
    "source_records_exact": 16,
    "reproducible_package_builds": 6,
    "domain_exact": 144,
    "teacher_agreement": 144,
    "namespace_route_exact": 144,
    "other_domain_target_answers": 0,
    "cpu_gpu_exact": 144,
    "english_immutable": 720,
    "english_core_target_answers": 0,
    "package_lifecycle_exact": 6,
    "targeted_corruptions_rejected": 6,
    "source_parameters_copied": 0,
    "receiver_training_steps": 0,
}


def _validate_binding(root: Path, item: dict[str, Any], name: str) -> None:
    target = (root / str(item.get("path", ""))).resolve()
    if (
        not target.is_file()
        or target.stat().st_size != item.get("bytes")
        or sha256_file(target) != item.get("sha256")
    ):
        raise R14Error(f"R25 prerequisite changed: {name}")


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R25 config unreadable") from exc
    if (
        value.get("format") != "abi-r25-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("implementation_freeze_commit", "")))
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("layercake_commit", "")))
        or value.get("namespaces") != list(NAMESPACES)
        or value.get("host_initializations") != list(HOST_INITIALIZATIONS)
        or value.get("build_initializations") != list(BUILD_INITIALIZATIONS)
        or value.get("gates") != GATES
        or value.get("new_source_calls_authorized") is not False
        or value.get("training_authorized") is not False
        or value.get("package_mutation_authorized") is not False
    ):
        raise R14Error("R25 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R25 ABI code inventory changed")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R25 ABI code changed: {relative}")
    layercake = value.get("layercake_code_sha256")
    if not isinstance(layercake, dict) or set(layercake) != set(LAYERCAKE_PATHS):
        raise R14Error("R25 LayerCake code inventory changed")
    layercake_root = (root / "../layercake_release").resolve()
    for relative, expected in layercake.items():
        target = layercake_root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R25 LayerCake code changed: {relative}")
    for name in (
        "r16_source_rows",
        "r16_receipt",
        "r16_strict_verification",
        "r23_config",
        "r23_reveal",
        "r23_live_verification",
        "r23_hidden_rows",
    ):
        _validate_binding(root, value.get(name, {}), name)
    for name in ("r16_packages", "english_packages"):
        items = value.get(name)
        expected_count = 2 if name == "r16_packages" else 6
        if not isinstance(items, list) or len(items) != expected_count:
            raise R14Error(f"R25 {name} inventory changed")
        for item in items:
            _validate_binding(root, item, name)
    r16 = json_object(root / value["r16_strict_verification"]["path"])
    r23 = json_object(root / value["r23_live_verification"]["path"])
    if (
        r16.get("verdict") != "PASS"
        or r16.get("facts_exact") != 16
        or r16.get("evaluation_rows_exact") != 48
        or r23.get("status") != "PASS_STRICTLY_VERIFIED_SEMANTIC_REPLICATION"
        or r23.get("evidence_sha256") != selfless_evidence_hash(r23)
    ):
        raise R14Error("R25 scientific prerequisite failed")
    source_splits(root / value["r16_source_rows"]["path"])
    return value
