"""Fail-closed binding for the R26 direct-import replication."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from experiments.canonical_layercake_import_r25.binding import load_config as load_r25
from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file
from experiments.generative_transfer_r21.hash_assurance_binding import selfless_evidence_hash

CODE_PATHS = (
    "experiments/factual_semantic_r16/public_sequence_scoring.py",
    "experiments/canonical_layercake_replication_r26/PROTOCOL.md",
    "experiments/canonical_layercake_replication_r26/prepare_source.py",
    "experiments/canonical_layercake_replication_r26/prepare_import_config.py",
    "experiments/canonical_layercake_replication_r26/binding.py",
    "experiments/canonical_layercake_replication_r26/freeze_config.py",
    "experiments/canonical_layercake_replication_r26/run.py",
    "experiments/canonical_layercake_replication_r26/verify.py",
)


def _binding(root: Path, item: dict[str, Any], name: str) -> None:
    target = root / str(item.get("path", ""))
    if (
        not target.is_file()
        or target.stat().st_size != item.get("bytes")
        or sha256_file(target) != item.get("sha256")
    ):
        raise R14Error(f"R26 prerequisite changed: {name}")


def load_config(root: Path, path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R26 config unreadable") from exc
    if (
        value.get("format") != "abi-r26-config/1"
        or not re.fullmatch(r"[0-9a-f]{40}", str(value.get("protocol_freeze_commit", "")))
        or value.get("candidate_changes_from_r25") != 0
        or value.get("gate_changes_from_r25") != 0
        or value.get("prospective_scorer") != "r16-normalized-text"
        or value.get("source_selection_generated_after_protocol_freeze") is not True
    ):
        raise R14Error("R26 governance changed")
    code = value.get("code_sha256")
    if not isinstance(code, dict) or set(code) != set(CODE_PATHS):
        raise R14Error("R26 code inventory changed")
    for relative, expected in code.items():
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise R14Error(f"R26 code changed: {relative}")
    for name in (
        "r25_implementation_config",
        "r25_posthoc_repair",
        "source_config",
        "source_reveal",
        "source_receipt",
        "source_strict_verification",
        "import_config",
    ):
        _binding(root, value.get(name, {}), name)
    source = json_object(root / value["source_receipt"]["path"])
    strict = json_object(root / value["source_strict_verification"]["path"])
    repair = json_object(root / value["r25_posthoc_repair"]["path"])
    if (
        source.get("verdict") != "PASS"
        or source.get("metrics", {}).get("evaluation_rows") != 48
        or strict.get("verdict") != "PASS"
        or strict.get("facts_exact") != 16
        or strict.get("evaluation_rows_exact") != 48
        or repair.get("status")
        != "PASS_POSTHOC_SCORER_REPAIR_REQUIRES_FRESH_REPLICATION"
        or repair.get("evidence_sha256") != selfless_evidence_hash(repair)
    ):
        raise R14Error("R26 scientific prerequisite failed")
    load_r25(root, root / value["import_config"]["path"])
    return value
