"""Fail-closed custody checks for the R11 execution boundary inherited by R12."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.native_isa_r11.core import sha256_file
from experiments.native_transfer_r8.capability_generator import canonical_json_bytes

from .teacher import R12TeacherError


def verify_r11_freeze(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every R11 binding instead of trusting a scientific boolean."""
    freeze = config.get("r11_freeze")
    if not isinstance(freeze, Mapping):
        raise R12TeacherError("missing R11 freeze specification")
    relative_manifest = freeze.get("binding_manifest")
    expected_manifest_sha = freeze.get("binding_manifest_sha256")
    if not isinstance(relative_manifest, str) or not isinstance(
        expected_manifest_sha, str
    ):
        raise R12TeacherError("invalid R11 binding manifest specification")
    manifest_path = root / relative_manifest
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha:
        raise R12TeacherError("sealed R11 binding manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise R12TeacherError("sealed R11 binding manifest has no bindings")
    verified: dict[str, str] = {}
    for relative_path, expected_sha in sorted(bindings.items()):
        if not isinstance(relative_path, str) or not isinstance(expected_sha, str):
            raise R12TeacherError("invalid sealed R11 binding")
        path = root / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise R12TeacherError(f"sealed R11 binding changed: {relative_path}")
        verified[relative_path] = expected_sha
    receipt = {
        "sealed_tag": freeze.get("sealed_tag"),
        "sealed_commit": freeze.get("sealed_commit"),
        "binding_manifest": relative_manifest,
        "binding_manifest_sha256": expected_manifest_sha,
        "verified_bindings": verified,
    }
    receipt["evidence_sha256"] = hashlib.sha256(
        canonical_json_bytes(receipt)
    ).hexdigest()
    return receipt
