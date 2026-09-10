"""Fail-closed bindings for the preregistered R18 hidden replication."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import R14Error, json_object, sha256_file


def load_bound_inputs(
    root: Path, config_path: Path, reveal_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json_object(config_path)
    reveal = json_object(reveal_path)
    if config.get("format") != "abi-r18-heldout-config/1":
        raise R14Error("R18 held-out config changed")
    code = config.get("code_sha256")
    if not isinstance(code, dict) or not code:
        raise R14Error("R18 held-out code inventory missing")
    for relative, expected in code.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise R14Error(f"R18 held-out code changed: {relative}")
    if (
        not reveal_path.is_file()
        or sha256_file(reveal_path) != config.get("reveal_sha256")
        or reveal.get("format") != "abi-r18-heldout-reveal/1"
        or reveal.get("commitment") != config.get("heldout_seed_commitment")
    ):
        raise R14Error("R18 held-out reveal changed")
    try:
        secret = bytes.fromhex(str(reveal["secret_hex"]))
    except (KeyError, ValueError) as exc:
        raise R14Error("R18 held-out secret changed") from exc
    if len(secret) != 32 or hashlib.sha256(secret).hexdigest() != reveal["commitment"]:
        raise R14Error("R18 held-out commitment mismatch")
    prerequisites = config.get("public_prerequisite")
    if not isinstance(prerequisites, dict):
        raise R14Error("R18 public prerequisite inventory missing")
    for name, item in prerequisites.items():
        if not isinstance(item, dict):
            raise R14Error(f"R18 public prerequisite changed: {name}")
        path = root / str(item.get("path"))
        if not path.is_file() or sha256_file(path) != item.get("sha256"):
            raise R14Error(f"R18 public prerequisite changed: {name}")
    return config, reveal
