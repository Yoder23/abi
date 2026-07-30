"""
ABI - Autonomous Basis Injection.

Published result: T5-large, Path 2C - NIB PASS, top-5 agreement = 0.8725.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "build_abi_artifact": ("abi.artifacts", "build_abi_artifact"),
    "build_compatibility_certificate": (
        "abi.artifacts",
        "build_compatibility_certificate",
    ),
    "build_cost_ledger": ("abi.artifacts", "build_cost_ledger"),
    "module_state_sha256": ("abi.artifacts", "module_state_sha256"),
    "build_capability_inventory": (
        "abi.capability_pipeline",
        "build_capability_inventory",
    ),
    "build_extraction_bundle": (
        "abi.capability_pipeline",
        "build_extraction_bundle",
    ),
    "build_nested_teacher_budgets": (
        "abi.capability_pipeline",
        "build_nested_teacher_budgets",
    ),
    "build_semantic_retention_certificate": (
        "abi.capability_pipeline",
        "build_semantic_retention_certificate",
    ),
    "build_source_model_manifest": (
        "abi.capability_pipeline",
        "build_source_model_manifest",
    ),
    "build_user_selection_plan": (
        "abi.capability_pipeline",
        "build_user_selection_plan",
    ),
    "verify_extraction_bundle": (
        "abi.capability_pipeline",
        "verify_extraction_bundle",
    ),
}

__all__ = [
    "build_abi_artifact",
    "build_compatibility_certificate",
    "build_cost_ledger",
    "module_state_sha256",
    "build_capability_inventory",
    "build_extraction_bundle",
    "build_nested_teacher_budgets",
    "build_semantic_retention_certificate",
    "build_source_model_manifest",
    "build_user_selection_plan",
    "verify_extraction_bundle",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
