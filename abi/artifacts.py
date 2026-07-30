"""Artifact and certificate helpers for ABI transfer runs.

These helpers keep provenance logic out of experiment scripts. They are small
on purpose: the output must be easy to audit in a result JSON.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn


SCHEMA_VERSION = "abi-artifact-v1"
CERTIFICATE_VERSION = "abi-compatibility-v1"
COST_LEDGER_VERSION = "abi-cost-ledger-v1"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    if value.dtype is torch.bfloat16:
        return value.view(torch.int16).numpy().tobytes()
    return value.numpy().tobytes()


def module_state_sha256(module: nn.Module) -> str:
    """Return a deterministic SHA-256 over a module state dict."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def module_param_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def trainable_param_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters() if param.requires_grad))


def build_abi_artifact(
    *,
    source_model: str,
    target_model: str,
    d_abi: int,
    domain_corpus: str,
    calibration_mode: str,
    calibration_init: str,
    source_domain_core_sha256: str,
    source_domain_full_sha256: str,
    rotated_core_initial_sha256: list[str],
    rotated_core_final_sha256: list[str],
    rotated_full_initial_sha256: list[str],
    rotated_full_final_sha256: list[str],
    copied_payload_core_params: int,
    copied_payload_full_params: int,
    trainable_groups: list[dict[str, Any]],
    alignment: dict[str, Any],
    target_side_components: list[str],
    oracle_mode: str = "full_native_target_oracle",
) -> dict[str, Any]:
    core_expected_frozen = calibration_mode in {
        "freeze_domain_net",
        "freeze_all_domain",
    }
    core_unchanged = rotated_core_initial_sha256 == rotated_core_final_sha256
    full_unchanged = rotated_full_initial_sha256 == rotated_full_final_sha256
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "rotated_source_domain_core",
        "source_model": source_model,
        "target_model": target_model,
        "d_abi": int(d_abi),
        "domain_corpus": domain_corpus,
        "calibration_mode": calibration_mode,
        "calibration_init": calibration_init,
        "oracle_mode": oracle_mode,
        "source_domain_core_sha256": source_domain_core_sha256,
        "source_domain_full_sha256": source_domain_full_sha256,
        "rotated_core_initial_sha256": rotated_core_initial_sha256,
        "rotated_core_final_sha256": rotated_core_final_sha256,
        "rotated_full_initial_sha256": rotated_full_initial_sha256,
        "rotated_full_final_sha256": rotated_full_final_sha256,
        "domain_core_frozen_claim": core_expected_frozen,
        "domain_core_unchanged_during_calibration": core_unchanged,
        "full_domain_module_unchanged_during_calibration": full_unchanged,
        "full_domain_module_change_allowed": calibration_mode
        in {"freeze_domain_net", "train_domain"},
        "copied_payload_core_params": int(copied_payload_core_params),
        "copied_payload_full_params": int(copied_payload_full_params),
        "rotation_member_count": len(rotated_core_initial_sha256),
        "alignment_method": alignment.get("align_map", alignment.get("mode")),
        "alignment_final_after_mean": alignment.get("final_after_mean"),
        "target_side_components": target_side_components,
        "target_side_trainable_groups": trainable_groups,
    }


def build_compatibility_certificate(
    *,
    artifact: dict[str, Any],
    alignment: dict[str, Any],
    nib_l2: dict[str, Any],
    posthoc_logit_scale: dict[str, Any],
    posthoc_logit_bias: dict[str, Any],
    target_native_oracle_required: bool,
    source_preservation: dict[str, Any] | None = None,
    selective_transfer: dict[str, Any] | None = None,
    oracle_mode: str = "full_native_target_oracle",
) -> dict[str, Any]:
    gates = {
        "alignment_measured": alignment.get("final_after_mean") is not None,
        "domain_core_freeze_verified": bool(
            artifact["domain_core_unchanged_during_calibration"]
        )
        if artifact["domain_core_frozen_claim"]
        else None,
        "target_reference_nib_pass": bool(nib_l2.get("pass", False)),
        "target_native_nib_pass": (
            bool(nib_l2.get("pass", False)) if target_native_oracle_required else None
        ),
        "source_preservation_measured": source_preservation is not None,
        "selective_transfer_measured": selective_transfer is not None,
        "off_domain_no_leakage_pass": (
            bool(selective_transfer.get("off_domain_no_leakage_pass", False))
            if selective_transfer is not None
            else None
        ),
        "selective_transfer_pass": (
            bool(selective_transfer.get("selective_transfer_pass", False))
            if selective_transfer is not None
            else None
        ),
        "oracle_light_mode": oracle_mode != "full_native_target_oracle",
    }
    claim_scope = (
        "scoped_target_native_oracle_transfer"
        if target_native_oracle_required
        else "oracle_light_transfer_probe"
    )
    return {
        "schema_version": CERTIFICATE_VERSION,
        "claim_scope": claim_scope,
        "oracle_mode": oracle_mode,
        "target_native_oracle_required": bool(target_native_oracle_required),
        "gates": gates,
        "alignment": alignment,
        "nib_l2": nib_l2,
        "source_preservation": source_preservation,
        "selective_transfer": selective_transfer,
        "posthoc_logit_scale": posthoc_logit_scale,
        "posthoc_logit_bias": posthoc_logit_bias,
        "claim_boundary": (
            "This certificate supports scoped ABI transfer against a target "
            "native oracle. It does not by itself prove lossless source-domain "
            "migration, selective off-domain noninterference, or production "
            "GPT5-to-GPT6 transfer."
        ),
    }


def build_cost_ledger(
    *,
    phase_seconds: dict[str, float],
    phase_steps: dict[str, int],
    token_counts: dict[str, int],
    param_counts: dict[str, int],
    repeat_count: int = 1,
    failed_search_runs_counted: int = 0,
) -> dict[str, Any]:
    total_seconds = float(sum(phase_seconds.values()))
    return {
        "schema_version": COST_LEDGER_VERSION,
        "phase_seconds": {
            key: round(float(value), 3) for key, value in phase_seconds.items()
        },
        "phase_steps": {key: int(value) for key, value in phase_steps.items()},
        "token_counts": {key: int(value) for key, value in token_counts.items()},
        "param_counts": {key: int(value) for key, value in param_counts.items()},
        "repeat_count": int(repeat_count),
        "failed_search_runs_counted": int(failed_search_runs_counted),
        "total_measured_seconds": round(total_seconds, 3),
        "cost_boundary": (
            "This ledger counts the measured phases in this run only. It does "
            "not include prior failed sweeps unless failed_search_runs_counted "
            "is nonzero."
        ),
    }
