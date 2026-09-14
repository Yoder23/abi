"""Immutable package boundary for the R91 structure-invariant bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import _sha256_file
from experiments.joint_span_r88.core_v1 import JointSpanBridge, bridge_parameter_count


PACKAGE_FORMAT = "abi-r91-structure-invariant-joint-span-package/1"
ARCHITECTURE = "structure-invariant-bidirectional-joint-span-selector/1"


class R91PackageError(RuntimeError):
    pass


def validate_metadata(metadata: dict[str, Any], root: Path) -> None:
    unsigned = dict(metadata)
    claimed = unsigned.pop("manifest_sha256", None)
    bridge = metadata.get("bridge", {})
    boundary = metadata.get("source_boundary", {})
    if (
        _manifest_sha(unsigned) != claimed
        or metadata.get("format") != PACKAGE_FORMAT
        or metadata.get("status") != "TRAINED_FROZEN_DEVELOPMENT_UNSCREENED"
        or bridge.get("architecture") != ARCHITECTURE
        or bridge.get("parameter_count") != bridge_parameter_count()
        or bridge.get("contains_prompt_templates") is not False
        or bridge.get("contains_rules") is not False
        or bridge.get("contains_output_lookup_table") is not False
        or boundary.get("teacher_present_during_training") is not False
        or boundary.get("teacher_present_at_inference") is not False
        or boundary.get("source_parameters_copied") != 0
        or boundary.get("source_logits_stored") != 0
        or boundary.get("source_hidden_activations_stored") != 0
        or metadata.get("checkpoint", {}).get("sha256")
        != _sha256_file(root / "bridge.safetensors")
    ):
        raise R91PackageError("R91 package metadata is invalid or stale")


def load_package(
    path: str | Path, *, device: str | torch.device
) -> tuple[JointSpanBridge, dict[str, Any]]:
    root = Path(path).resolve()
    checkpoint = root / "bridge.safetensors"
    metadata_path = root / "metadata.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise R91PackageError("R91 package files are absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, root)
    bridge = JointSpanBridge().to(device)
    bridge.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    bridge.eval()
    return bridge, metadata
