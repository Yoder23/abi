"""Package boundary for the R96 length-invariant neural span bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from abi.layercake_full_core_acquisition import _manifest_sha
from abi.layercake_host import _sha256_file
from experiments.joint_span_r88.core_v1 import JointSpanBridge, bridge_parameter_count


PACKAGE_FORMAT = "abi-r96-length-invariant-joint-span-package/1"
ARCHITECTURE = "length-and-structure-invariant-bidirectional-joint-span-selector/1"


class R96PackageError(RuntimeError):
    pass


def validate_metadata(metadata: dict[str, Any], root: Path) -> None:
    unsigned = dict(metadata)
    claimed = unsigned.pop("manifest_sha256", None)
    bridge = metadata.get("bridge", {})
    boundary = metadata.get("source_boundary", {})
    init = metadata.get("initialization", {})
    if (
        _manifest_sha(unsigned) != claimed
        or metadata.get("format") != PACKAGE_FORMAT
        or metadata.get("status") != "TRAINED_FROZEN_DEVELOPMENT_UNSCREENED"
        or bridge.get("architecture") != ARCHITECTURE
        or bridge.get("parameter_count") != bridge_parameter_count()
        or bridge.get("contains_prompt_templates") is not False
        or bridge.get("contains_rules") is not False
        or bridge.get("contains_output_lookup_table") is not False
        or init.get("r91_checkpoint_sha256") != "04b350e0a7f22ac4facb380f6bf99edfc0022697a312248972491434234a4fba"
        or boundary.get("teacher_present_during_training") is not False
        or boundary.get("teacher_present_at_inference") is not False
        or boundary.get("source_parameters_copied") != 0
        or boundary.get("source_logits_stored") != 0
        or boundary.get("source_hidden_activations_stored") != 0
        or metadata.get("checkpoint", {}).get("sha256") != _sha256_file(root / "bridge.safetensors")
    ):
        raise R96PackageError("R96 package metadata is invalid or stale")


def load_package(path: str | Path, *, device: str | torch.device) -> tuple[JointSpanBridge, dict[str, Any]]:
    root = Path(path).resolve()
    checkpoint, metadata_path = root / "bridge.safetensors", root / "metadata.json"
    if not checkpoint.is_file() or not metadata_path.is_file():
        raise R96PackageError("R96 package files are absent")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_metadata(metadata, root)
    bridge = JointSpanBridge().to(device)
    bridge.load_state_dict(load_file(str(checkpoint), device=str(device)), strict=True)
    bridge.eval()
    return bridge, metadata
