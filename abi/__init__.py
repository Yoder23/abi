"""ABI's supported capability-acquisition and segregation API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.4.0a1"


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
    "build_core_domain_segregation_manifest": (
        "abi.capability_segregation",
        "build_core_domain_segregation_manifest",
    ),
    "build_domain_ontology": (
        "abi.capability_segregation",
        "build_domain_ontology",
    ),
    "build_segregated_extraction_record": (
        "abi.capability_segregation",
        "build_segregated_extraction_record",
    ),
    "validate_core_domain_segregation_manifest": (
        "abi.capability_segregation",
        "validate_core_domain_segregation_manifest",
    ),
    "validate_domain_ontology": (
        "abi.capability_segregation",
        "validate_domain_ontology",
    ),
}

__all__ = [
    "__version__",
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
    "build_core_domain_segregation_manifest",
    "build_domain_ontology",
    "build_segregated_extraction_record",
    "validate_core_domain_segregation_manifest",
    "validate_domain_ontology",
]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
