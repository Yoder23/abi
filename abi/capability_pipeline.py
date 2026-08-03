"""Content-addressed capability discovery and extraction for LayerCake.

The ABI research modules in this repository migrate frozen modules between
transformer backbones.  This module defines a different boundary: a frozen
source model is inspected and probed, its observed behavior is labeled, a user
selects English and/or specialist domains, and the selected records are packed
as an immutable *training artifact*.  The artifact is never a deployed cake.

The implementation is intentionally fail-closed:

* a capability inventory is bounded by a declared probe catalog;
* domain discovery is never described as exhaustive;
* all source weights, outputs, and token counts have immutable provenance;
* larger teacher-token budgets are nested supersets of smaller budgets;
* exact byte identity and bounded semantic retention are separate claims; and
* an extraction bundle explicitly declares that it contains teacher material
  and must not be installed as a LayerCake capability package.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import zipfile
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .capability_segregation import (
    SEGREGATED_RECORD_SCHEMA,
    validate_core_domain_segregation_manifest,
    validate_domain_ontology,
)
from .layercake_acquisition import (
    ENGLISH_CORE_CAPABILITIES,
    AcquisitionAccountingError,
    validate_labeled_extraction_record,
)


SOURCE_MANIFEST_SCHEMA = "abi-source-model-manifest/1"
PROBE_RESULT_SCHEMA = "abi-capability-probe-result/1"
INVENTORY_SCHEMA = "abi-probe-bounded-capability-inventory/1"
SELECTION_SCHEMA = "abi-user-capability-selection/1"
BUDGET_MANIFEST_SCHEMA = "abi-nested-teacher-budget-manifest/1"
BUNDLE_MANIFEST_SCHEMA = "abi-capability-extraction-bundle/1"
SURVEY_ARTIFACT_ROLE = "source_capability_survey_vault"
LEGACY_TRAINING_ARTIFACT_ROLE = "selected_layercake_training_material"
TRAINING_ARTIFACT_ROLE = "selected_layercake_training_material_v2"
SEGREGATED_TRAINING_ARTIFACT_ROLE = (
    "segregated_layercake_training_material_v3"
)
RETENTION_CERTIFICATE_SCHEMA = "abi-layercake-retention-certificate/1"

_BUNDLE_MEMBERS = frozenset(
    {
        "manifest.json",
        "sources.json",
        "records.jsonl",
        "probe_results.json",
        "inventory.json",
        "selection.json",
        "budgets.json",
        "ledger.json",
    }
)
_SEGREGATED_BUNDLE_MEMBERS = _BUNDLE_MEMBERS | {
    "domain_ontology.json",
    "segregation.json",
}


class CapabilityPipelineError(AcquisitionAccountingError):
    """Raised when capability evidence or an extraction bundle is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value in the repository's canonical form."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping_hash(value: Mapping[str, Any], hash_field: str) -> str:
    body = dict(value)
    body.pop(hash_field, None)
    return sha256_bytes(canonical_json_bytes(body))


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilityPipelineError(f"{name} must be a non-empty string")
    return value


def _require_sha256(name: str, value: Any) -> str:
    value = _require_string(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CapabilityPipelineError(f"{name} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CapabilityPipelineError(f"{name} must be a non-negative integer")
    return value


def _safe_relative_path(name: str, value: Any) -> str:
    raw = _require_string(name, value).replace("\\", "/")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise CapabilityPipelineError(f"{name} must be a safe relative path")
    return candidate.as_posix()


def _validate_self_hash(
    value: Mapping[str, Any],
    *,
    schema: str,
    schema_field: str,
    hash_field: str,
) -> None:
    if value.get(schema_field) != schema:
        raise CapabilityPipelineError(f"unsupported {schema_field}")
    expected = _mapping_hash(value, hash_field)
    if value.get(hash_field) != expected:
        raise CapabilityPipelineError(f"stale or invalid {hash_field}")


def build_source_model_manifest(
    *,
    model_id: str,
    revision: str,
    revision_is_immutable: bool,
    architecture: str,
    parameter_count: int,
    tokenizer_id: str,
    tokenizer_revision: str,
    license_id: str,
    weight_files: Sequence[Mapping[str, Any]],
    source_kind: str = "huggingface_causal_lm",
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Describe the exact frozen source model used for extraction.

    ``weight_files`` entries must contain ``relative_path``, ``sha256``, and
    ``bytes``.  A mutable branch name such as ``main`` may be recorded for
    diagnostics, but it cannot produce a promotion-eligible source manifest.
    """

    if not isinstance(revision_is_immutable, bool):
        raise CapabilityPipelineError("revision_is_immutable must be boolean")
    if not isinstance(trust_remote_code, bool):
        raise CapabilityPipelineError("trust_remote_code must be boolean")
    if not weight_files:
        raise CapabilityPipelineError("at least one source weight file is required")
    normalized_weights: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw in weight_files:
        path = _safe_relative_path("weight relative_path", raw.get("relative_path"))
        if path in paths:
            raise CapabilityPipelineError(f"duplicate source weight path: {path}")
        paths.add(path)
        normalized_weights.append(
            {
                "relative_path": path,
                "sha256": _require_sha256("weight sha256", raw.get("sha256")),
                "bytes": _require_nonnegative_int("weight bytes", raw.get("bytes")),
            }
        )
    normalized_weights.sort(key=lambda row: row["relative_path"])
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA,
        "model_id": _require_string("model_id", model_id),
        "revision": _require_string("revision", revision),
        "revision_is_immutable": revision_is_immutable,
        "source_kind": _require_string("source_kind", source_kind),
        "architecture": _require_string("architecture", architecture),
        "parameter_count": _require_nonnegative_int("parameter_count", parameter_count),
        "tokenizer_id": _require_string("tokenizer_id", tokenizer_id),
        "tokenizer_revision": _require_string(
            "tokenizer_revision", tokenizer_revision
        ),
        "license_id": _require_string("license_id", license_id),
        "trust_remote_code": trust_remote_code,
        "weight_files": normalized_weights,
        "weight_file_count": len(normalized_weights),
        "weight_bytes": sum(row["bytes"] for row in normalized_weights),
        "promotion_eligible": revision_is_immutable and not trust_remote_code,
    }
    manifest["source_manifest_sha256"] = _mapping_hash(
        manifest, "source_manifest_sha256"
    )
    return manifest


def validate_source_model_manifest(manifest: Mapping[str, Any]) -> None:
    """Rebuild and validate a source manifest."""

    rebuilt = build_source_model_manifest(
        model_id=manifest.get("model_id"),
        revision=manifest.get("revision"),
        revision_is_immutable=manifest.get("revision_is_immutable"),
        architecture=manifest.get("architecture"),
        parameter_count=manifest.get("parameter_count"),
        tokenizer_id=manifest.get("tokenizer_id"),
        tokenizer_revision=manifest.get("tokenizer_revision"),
        license_id=manifest.get("license_id"),
        weight_files=manifest.get("weight_files"),
        source_kind=manifest.get("source_kind"),
        trust_remote_code=manifest.get("trust_remote_code"),
    )
    if dict(manifest) != rebuilt:
        raise CapabilityPipelineError("source model manifest is stale or invalid")


def build_probe_result(
    *,
    record: Mapping[str, Any],
    source_manifest_sha256: str,
    probe_id: str,
    evaluator: Mapping[str, Any],
    passed: bool,
    score: float,
    seed: int,
) -> dict[str, Any]:
    """Bind one labeled extraction record to its deterministic probe result."""

    validate_labeled_extraction_record(record)
    if not isinstance(evaluator, Mapping) or not evaluator:
        raise CapabilityPipelineError("evaluator must be a non-empty object")
    if not isinstance(passed, bool):
        raise CapabilityPipelineError("passed must be boolean")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise CapabilityPipelineError("score must be numeric")
    score = float(score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise CapabilityPipelineError("score must be finite and in [0, 1]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CapabilityPipelineError("seed must be a non-negative integer")
    result = {
        "schema_version": PROBE_RESULT_SCHEMA,
        "record_id": record["record_id"],
        "source_manifest_sha256": _require_sha256(
            "source_manifest_sha256", source_manifest_sha256
        ),
        "probe_id": _require_string("probe_id", probe_id),
        "destination_scope": record["destination_scope"],
        "capability": record["capability"],
        "domain": record["domain"],
        "split": record["split"],
        "evaluator": dict(evaluator),
        "passed": passed,
        "score": score,
        "seed": seed,
    }
    result["probe_result_sha256"] = _mapping_hash(
        result, "probe_result_sha256"
    )
    return result


def validate_probe_result(result: Mapping[str, Any]) -> None:
    _validate_self_hash(
        result,
        schema=PROBE_RESULT_SCHEMA,
        schema_field="schema_version",
        hash_field="probe_result_sha256",
    )
    _require_sha256("record_id", result.get("record_id"))
    _require_sha256(
        "source_manifest_sha256", result.get("source_manifest_sha256")
    )
    _require_string("probe_id", result.get("probe_id"))
    if result.get("destination_scope") not in {"english_core", "domain_cake"}:
        raise CapabilityPipelineError("invalid probe destination_scope")
    _require_string("capability", result.get("capability"))
    _require_string("domain", result.get("domain"))
    if result.get("split") not in {"search", "validation", "final_test"}:
        raise CapabilityPipelineError("invalid probe split")
    if not isinstance(result.get("evaluator"), Mapping) or not result["evaluator"]:
        raise CapabilityPipelineError("invalid probe evaluator")
    if not isinstance(result.get("passed"), bool):
        raise CapabilityPipelineError("probe passed must be boolean")
    score = result.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise CapabilityPipelineError("invalid probe score")


def _wilson_lower(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = proportion + z * z / (2.0 * total)
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * total)) / total
    )
    return max(0.0, (center - margin) / denominator)


def build_capability_inventory(
    *,
    source_manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    probe_results: Sequence[Mapping[str, Any]],
    minimum_distinct_probes: int = 20,
    minimum_pass_rate: float = 0.90,
    minimum_wilson_lower_bound: float = 0.75,
    qualification_splits: Sequence[str] = ("validation",),
) -> dict[str, Any]:
    """Create a probe-bounded, non-exhaustive capability inventory."""

    validate_source_model_manifest(source_manifest)
    minimum_distinct_probes = _require_nonnegative_int(
        "minimum_distinct_probes", minimum_distinct_probes
    )
    if minimum_distinct_probes < 1:
        raise CapabilityPipelineError("minimum_distinct_probes must be positive")
    for name, value in {
        "minimum_pass_rate": minimum_pass_rate,
        "minimum_wilson_lower_bound": minimum_wilson_lower_bound,
    }.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise CapabilityPipelineError(f"{name} must be in [0, 1]")
    normalized_qualification_splits = sorted(set(qualification_splits))
    if (
        not normalized_qualification_splits
        or any(
            split not in {"search", "validation"}
            for split in normalized_qualification_splits
        )
    ):
        raise CapabilityPipelineError(
            "qualification_splits must use search and/or validation, never final_test"
        )

    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        validate_labeled_extraction_record(record)
        record_id = str(record["record_id"])
        if record_id in records_by_id:
            raise CapabilityPipelineError(f"duplicate record_id: {record_id}")
        if (
            record["source_model"] != source_manifest["model_id"]
            or record["source_model_revision"] != source_manifest["revision"]
        ):
            raise CapabilityPipelineError(
                "record source identity does not match the source manifest"
            )
        records_by_id[record_id] = record
    if not records_by_id:
        raise CapabilityPipelineError("inventory requires extraction records")

    result_hashes: set[str] = set()
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for result in probe_results:
        validate_probe_result(result)
        result_hash = str(result["probe_result_sha256"])
        if result_hash in result_hashes:
            raise CapabilityPipelineError(
                f"duplicate probe_result_sha256: {result_hash}"
            )
        result_hashes.add(result_hash)
        if result["source_manifest_sha256"] != source_manifest["source_manifest_sha256"]:
            raise CapabilityPipelineError("probe result names a different source manifest")
        record = records_by_id.get(str(result["record_id"]))
        if record is None:
            raise CapabilityPipelineError("probe result references an unknown record")
        for field in ("destination_scope", "capability", "domain", "split"):
            if result[field] != record[field]:
                raise CapabilityPipelineError(
                    f"probe result and record disagree on {field}"
                )
        groups[
            (
                str(result["destination_scope"]),
                str(result["domain"]),
                str(result["capability"]),
            )
        ].append(result)
    if not groups:
        raise CapabilityPipelineError("inventory requires probe results")

    entries: list[dict[str, Any]] = []
    for (scope, domain, capability), rows in sorted(groups.items()):
        qualifying_rows = [
            row for row in rows if row["split"] in normalized_qualification_splits
        ]
        distinct_probe_ids = {str(row["probe_id"]) for row in qualifying_rows}
        successes = sum(1 for row in qualifying_rows if row["passed"])
        total = len(qualifying_rows)
        pass_rate = successes / total if total else 0.0
        lower_bound = _wilson_lower(successes, total)
        available = (
            len(distinct_probe_ids) >= minimum_distinct_probes
            and pass_rate >= float(minimum_pass_rate)
            and lower_bound >= float(minimum_wilson_lower_bound)
        )
        entries.append(
            {
                "destination_scope": scope,
                "domain": domain,
                "capability": capability,
                "probe_count": len(rows),
                "qualification_probe_count": total,
                "distinct_probe_count": len(distinct_probe_ids),
                "successes": successes,
                "pass_rate": round(pass_rate, 8),
                "wilson_95_lower_bound": round(lower_bound, 8),
                "available": available,
                "status": "AVAILABLE" if available else "INSUFFICIENT_OR_FAILED",
                "probe_result_sha256": sorted(
                    str(row["probe_result_sha256"]) for row in rows
                ),
            }
        )

    inventory = {
        "schema_version": INVENTORY_SCHEMA,
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "source_model": source_manifest["model_id"],
        "source_model_revision": source_manifest["revision"],
        "catalog_scope": "probe_defined_not_exhaustive",
        "exhaustive_domain_discovery_claimed": False,
        "thresholds": {
            "minimum_distinct_probes": minimum_distinct_probes,
            "minimum_pass_rate": float(minimum_pass_rate),
            "minimum_wilson_95_lower_bound": float(
                minimum_wilson_lower_bound
            ),
            "qualification_splits": normalized_qualification_splits,
        },
        "entries": entries,
        "available_entry_count": sum(1 for entry in entries if entry["available"]),
    }
    inventory["inventory_sha256"] = _mapping_hash(inventory, "inventory_sha256")
    return inventory


def validate_capability_inventory(inventory: Mapping[str, Any]) -> None:
    _validate_self_hash(
        inventory,
        schema=INVENTORY_SCHEMA,
        schema_field="schema_version",
        hash_field="inventory_sha256",
    )
    _require_sha256(
        "source_manifest_sha256", inventory.get("source_manifest_sha256")
    )
    if inventory.get("catalog_scope") != "probe_defined_not_exhaustive":
        raise CapabilityPipelineError("inventory must declare its probe-bounded scope")
    if inventory.get("exhaustive_domain_discovery_claimed") is not False:
        raise CapabilityPipelineError("inventory may not claim exhaustive discovery")
    entries = inventory.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CapabilityPipelineError("inventory entries are missing")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise CapabilityPipelineError("inventory entry must be an object")
        if entry.get("destination_scope") not in {"english_core", "domain_cake"}:
            raise CapabilityPipelineError("invalid inventory destination_scope")
        _require_string("inventory domain", entry.get("domain"))
        _require_string("inventory capability", entry.get("capability"))
        if not isinstance(entry.get("available"), bool):
            raise CapabilityPipelineError("inventory availability must be boolean")


def _entry_rank(entry: Mapping[str, Any]) -> tuple[float, float, int, str]:
    return (
        float(entry.get("wilson_95_lower_bound", 0.0)),
        float(entry.get("pass_rate", 0.0)),
        int(entry.get("distinct_probe_count", 0)),
        str(entry.get("source_manifest_sha256", "")),
    )


def build_user_selection_plan(
    inventories: Sequence[Mapping[str, Any]],
    *,
    include_english_core: bool,
    english_capabilities: Sequence[str] | None = None,
    domains: Sequence[str],
    source_policy: str = "best_evidence",
    allow_unverified_development_selection: bool = False,
) -> dict[str, Any]:
    """Select English capabilities and domain cakes from one or many sources."""

    if not isinstance(include_english_core, bool):
        raise CapabilityPipelineError("include_english_core must be boolean")
    if not isinstance(allow_unverified_development_selection, bool):
        raise CapabilityPipelineError(
            "allow_unverified_development_selection must be boolean"
        )
    if source_policy not in {"best_evidence", "all_qualified_sources"}:
        raise CapabilityPipelineError("unsupported source_policy")
    normalized_domains = sorted(
        {_require_string("domain", domain) for domain in domains}
    )
    if english_capabilities is not None and not include_english_core:
        raise CapabilityPipelineError(
            "English capability subset requires include_english_core"
        )
    requested_english_capabilities = (
        sorted(ENGLISH_CORE_CAPABILITIES)
        if include_english_core and english_capabilities is None
        else sorted(
            {
                _require_string("English capability", capability)
                for capability in (english_capabilities or ())
            }
        )
    )
    unsupported_english = sorted(
        set(requested_english_capabilities) - set(ENGLISH_CORE_CAPABILITIES)
    )
    if unsupported_english:
        raise CapabilityPipelineError(
            f"unsupported English capabilities: {unsupported_english}"
        )
    if include_english_core and not requested_english_capabilities:
        raise CapabilityPipelineError(
            "English capability selection cannot be empty"
        )
    if not include_english_core and not normalized_domains:
        raise CapabilityPipelineError("selection must request English or a domain")

    normalized: list[dict[str, Any]] = []
    inventory_hashes: set[str] = set()
    for inventory in inventories:
        validate_capability_inventory(inventory)
        inventory_hash = str(inventory["inventory_sha256"])
        if inventory_hash in inventory_hashes:
            raise CapabilityPipelineError(f"duplicate inventory: {inventory_hash}")
        inventory_hashes.add(inventory_hash)
        for entry in inventory["entries"]:
            row = dict(entry)
            row["source_manifest_sha256"] = inventory["source_manifest_sha256"]
            row["source_model"] = inventory["source_model"]
            row["source_model_revision"] = inventory["source_model_revision"]
            row["inventory_sha256"] = inventory_hash
            normalized.append(row)
    if not normalized:
        raise CapabilityPipelineError("no inventories supplied")

    def eligible(row: Mapping[str, Any]) -> bool:
        return bool(row["available"]) or allow_unverified_development_selection

    selected: list[dict[str, Any]] = []
    if include_english_core:
        for capability in requested_english_capabilities:
            candidates = [
                row
                for row in normalized
                if row["destination_scope"] == "english_core"
                and row["domain"] == "domain_independent"
                and row["capability"] == capability
                and eligible(row)
            ]
            if not candidates:
                raise CapabilityPipelineError(
                    f"no qualified source for English capability: {capability}"
                )
            candidates.sort(key=_entry_rank, reverse=True)
            chosen = (
                candidates
                if source_policy == "all_qualified_sources"
                else candidates[:1]
            )
            selected.extend(chosen)

    for domain in normalized_domains:
        candidates = [
            row
            for row in normalized
            if row["destination_scope"] == "domain_cake"
            and row["domain"] == domain
            and eligible(row)
        ]
        if not candidates:
            raise CapabilityPipelineError(
                f"no qualified source for requested domain: {domain}"
            )
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            by_source[str(row["source_manifest_sha256"])].append(row)
        source_ranks = sorted(
            by_source,
            key=lambda source_hash: min(
                (_entry_rank(row) for row in by_source[source_hash]),
                default=(0.0, 0.0, 0, source_hash),
            ),
            reverse=True,
        )
        chosen_sources = (
            source_ranks if source_policy == "all_qualified_sources" else source_ranks[:1]
        )
        for source_hash in chosen_sources:
            selected.extend(by_source[source_hash])

    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in selected:
        key = (
            str(row["destination_scope"]),
            str(row["domain"]),
            str(row["capability"]),
            str(row["source_manifest_sha256"]),
        )
        unique[key] = {
            "destination_scope": row["destination_scope"],
            "domain": row["domain"],
            "capability": row["capability"],
            "source_manifest_sha256": row["source_manifest_sha256"],
            "source_model": row["source_model"],
            "source_model_revision": row["source_model_revision"],
            "inventory_sha256": row["inventory_sha256"],
            "evidence_status": row["status"],
            "available": row["available"],
        }
    items = [unique[key] for key in sorted(unique)]
    plan = {
        "schema_version": SELECTION_SCHEMA,
        "include_english_core": include_english_core,
        "english_selection_scope": (
            "complete_core"
            if (
                include_english_core
                and set(requested_english_capabilities)
                == set(ENGLISH_CORE_CAPABILITIES)
            )
            else (
                "capability_subset"
                if include_english_core
                else "not_requested"
            )
        ),
        "requested_english_capabilities": (
            requested_english_capabilities
        ),
        "requested_domains": normalized_domains,
        "source_policy": source_policy,
        "allow_unverified_development_selection": (
            allow_unverified_development_selection
        ),
        "promotion_eligible": (
            not allow_unverified_development_selection
            and all(item["available"] for item in items)
            and (
                not include_english_core
                or set(requested_english_capabilities)
                == set(ENGLISH_CORE_CAPABILITIES)
            )
        ),
        "inventory_sha256": sorted(inventory_hashes),
        "selected_items": items,
        "selected_source_manifest_sha256": sorted(
            {str(item["source_manifest_sha256"]) for item in items}
        ),
    }
    plan["selection_sha256"] = _mapping_hash(plan, "selection_sha256")
    return plan


def validate_user_selection_plan(plan: Mapping[str, Any]) -> None:
    _validate_self_hash(
        plan,
        schema=SELECTION_SCHEMA,
        schema_field="schema_version",
        hash_field="selection_sha256",
    )
    if not isinstance(plan.get("selected_items"), list) or not plan["selected_items"]:
        raise CapabilityPipelineError("selection has no selected_items")
    if not isinstance(plan.get("promotion_eligible"), bool):
        raise CapabilityPipelineError("selection promotion_eligible must be boolean")
    for item in plan["selected_items"]:
        if not isinstance(item, Mapping):
            raise CapabilityPipelineError("selected item must be an object")
        _require_sha256(
            "selected source_manifest_sha256",
            item.get("source_manifest_sha256"),
        )


def build_inventory_survey_plan(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind every catalog entry to a source-survey vault.

    A survey is evidence collection, not a user deployment selection. Partial
    English catalogs and failed entries therefore remain representable without
    pretending that a complete English core has been qualified.
    """

    validate_capability_inventory(inventory)
    inventory_hash = str(inventory["inventory_sha256"])
    items = [
        {
            "destination_scope": entry["destination_scope"],
            "domain": entry["domain"],
            "capability": entry["capability"],
            "source_manifest_sha256": inventory["source_manifest_sha256"],
            "source_model": inventory["source_model"],
            "source_model_revision": inventory["source_model_revision"],
            "inventory_sha256": inventory_hash,
            "evidence_status": entry["status"],
            "available": entry["available"],
        }
        for entry in inventory["entries"]
    ]
    plan = {
        "schema_version": SELECTION_SCHEMA,
        "selection_purpose": "source_capability_survey",
        "include_english_core": False,
        "english_selection_scope": "not_requested",
        "requested_english_capabilities": [],
        "requested_domains": [],
        "source_policy": "survey_all_catalog_entries",
        "allow_unverified_development_selection": True,
        "promotion_eligible": False,
        "inventory_sha256": [inventory_hash],
        "selected_items": items,
        "selected_source_manifest_sha256": [
            inventory["source_manifest_sha256"]
        ],
    }
    plan["selection_sha256"] = _mapping_hash(plan, "selection_sha256")
    return plan


def records_for_selection(
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    *,
    split: str | None = None,
    allow_empty: bool = False,
) -> list[dict[str, Any]]:
    """Return records whose destination and immutable source match a plan."""

    validate_user_selection_plan(plan)
    if split is not None and split not in {"search", "validation", "final_test"}:
        raise CapabilityPipelineError("invalid split")
    selected_keys = {
        (
            item["destination_scope"],
            item["domain"],
            item["capability"],
            item["source_model"],
            item["source_model_revision"],
        )
        for item in plan["selected_items"]
    }
    selected_records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for record in records:
        validate_labeled_extraction_record(record)
        key = (
            record["destination_scope"],
            record["domain"],
            record["capability"],
            record["source_model"],
            record["source_model_revision"],
        )
        if key not in selected_keys or (split is not None and record["split"] != split):
            continue
        if record["record_id"] in record_ids:
            raise CapabilityPipelineError(
                f"duplicate selected record_id: {record['record_id']}"
            )
        record_ids.add(str(record["record_id"]))
        selected_records.append(dict(record))
    if not selected_records and not allow_empty:
        raise CapabilityPipelineError("selection produced no extraction records")
    return sorted(selected_records, key=lambda row: row["record_id"])


def build_nested_teacher_budgets(
    records: Sequence[Mapping[str, Any]],
    *,
    requested_teacher_token_budgets: Sequence[int],
    split: str,
    ordering_seed: str,
) -> list[dict[str, Any]]:
    """Create deterministic, capability-stratified nested record budgets."""

    if split not in {"search", "validation"}:
        raise CapabilityPipelineError(
            "nested budget construction is limited to search or validation"
        )
    ordering_seed = _require_string("ordering_seed", ordering_seed)
    budgets = [
        _require_nonnegative_int("teacher token budget", value)
        for value in requested_teacher_token_budgets
    ]
    if not budgets or any(value <= 0 for value in budgets):
        raise CapabilityPipelineError("teacher token budgets must be positive")
    if budgets != sorted(set(budgets)):
        raise CapabilityPipelineError(
            "teacher token budgets must be strictly increasing"
        )

    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for record in records:
        validate_labeled_extraction_record(record)
        if record["split"] != split:
            continue
        record_id = str(record["record_id"])
        if record_id in seen:
            raise CapabilityPipelineError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        key = (
            str(record["destination_scope"]),
            str(record["domain"]),
            str(record["capability"]),
        )
        strata[key].append(dict(record))
    if not strata:
        raise CapabilityPipelineError(f"no records for split {split}")
    for rows in strata.values():
        rows.sort(
            key=lambda row: sha256_bytes(
                f"{ordering_seed}:{row['record_id']}".encode("utf-8")
            )
        )
    queues = {key: deque(rows) for key, rows in sorted(strata.items())}
    order: list[dict[str, Any]] = []
    while any(queues.values()):
        for key in sorted(queues):
            if queues[key]:
                order.append(queues[key].popleft())

    manifests: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    next_index = 0
    token_total = 0
    prior_ids: set[str] = set()
    for requested in budgets:
        while next_index < len(order):
            candidate = order[next_index]
            candidate_tokens = int(candidate["teacher_tokens"])
            if token_total + candidate_tokens > requested:
                break
            prefix.append(candidate)
            token_total += candidate_tokens
            next_index += 1
        if not prefix:
            raise CapabilityPipelineError(
                f"budget {requested} cannot contain the first complete record"
            )
        ids = [str(record["record_id"]) for record in prefix]
        current_ids = set(ids)
        if not prior_ids.issubset(current_ids):
            raise CapabilityPipelineError("internal non-nested budget construction")
        manifest = {
            "schema_version": BUDGET_MANIFEST_SCHEMA,
            "budget_id": f"{split}-teacher-tokens-{requested}",
            "split": split,
            "requested_teacher_tokens": requested,
            "teacher_tokens": token_total,
            "record_count": len(ids),
            "record_ids": ids,
            "ordering_seed": ordering_seed,
            "nested": True,
        }
        manifest["budget_manifest_sha256"] = _mapping_hash(
            manifest, "budget_manifest_sha256"
        )
        manifests.append(manifest)
        prior_ids = current_ids
    return manifests


def _validate_budget_manifest(manifest: Mapping[str, Any]) -> None:
    _validate_self_hash(
        manifest,
        schema=BUDGET_MANIFEST_SCHEMA,
        schema_field="schema_version",
        hash_field="budget_manifest_sha256",
    )
    if manifest.get("split") not in {"search", "validation"}:
        raise CapabilityPipelineError("invalid budget split")
    ids = manifest.get("record_ids")
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) != len(set(ids))
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in ids
        )
    ):
        raise CapabilityPipelineError("invalid budget record_ids")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _zip_bytes(members: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, members[name])
    return output.getvalue()


def build_extraction_bundle(
    output_path: str | Path,
    *,
    source_manifests: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    probe_results: Sequence[Mapping[str, Any]],
    inventories: Sequence[Mapping[str, Any]],
    selection: Mapping[str, Any],
    budgets: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    artifact_role: str = SURVEY_ARTIFACT_ROLE,
    domain_ontology: Mapping[str, Any] | None = None,
    segregation_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a deterministic extraction bundle and return its exact identity."""

    if not source_manifests:
        raise CapabilityPipelineError("bundle requires source manifests")
    if artifact_role not in {
        SURVEY_ARTIFACT_ROLE,
        TRAINING_ARTIFACT_ROLE,
        SEGREGATED_TRAINING_ARTIFACT_ROLE,
    }:
        raise CapabilityPipelineError("unsupported extraction artifact_role")
    source_hashes: set[str] = set()
    validated_sources: list[dict[str, Any]] = []
    for source in source_manifests:
        validate_source_model_manifest(source)
        source_hash = str(source["source_manifest_sha256"])
        if source_hash in source_hashes:
            raise CapabilityPipelineError(f"duplicate source manifest: {source_hash}")
        source_hashes.add(source_hash)
        validated_sources.append(dict(source))
    validated_sources.sort(key=lambda row: row["source_manifest_sha256"])
    validated_records: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for record in records:
        validate_labeled_extraction_record(record)
        if record["record_id"] in record_ids:
            raise CapabilityPipelineError(f"duplicate record_id: {record['record_id']}")
        record_ids.add(str(record["record_id"]))
        validated_records.append(dict(record))
    validated_records.sort(key=lambda row: row["record_id"])
    if not validated_records:
        raise CapabilityPipelineError("bundle requires records")
    final_test_record_count = sum(
        1 for record in validated_records if record["split"] == "final_test"
    )
    non_search_record_count = sum(
        1 for record in validated_records if record["split"] != "search"
    )
    training_roles = {
        TRAINING_ARTIFACT_ROLE,
        SEGREGATED_TRAINING_ARTIFACT_ROLE,
    }
    if artifact_role in training_roles and non_search_record_count:
        raise CapabilityPipelineError(
            "selected LayerCake training material may contain search records only"
        )
    if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE:
        if domain_ontology is None or segregation_manifest is None:
            raise CapabilityPipelineError(
                "segregated training material requires ontology and purity manifest"
            )
        try:
            validate_domain_ontology(domain_ontology)
            validate_core_domain_segregation_manifest(
                segregation_manifest,
                validated_records,
                domain_ontology=domain_ontology,
            )
        except AcquisitionAccountingError as exc:
            raise CapabilityPipelineError(
                f"core/domain segregation gate failed: {exc}"
            ) from exc
        if any(
            record.get("schema_version") != SEGREGATED_RECORD_SCHEMA
            for record in validated_records
        ):
            raise CapabilityPipelineError(
                "segregated bundle contains a legacy extraction record"
            )
    elif domain_ontology is not None or segregation_manifest is not None:
        raise CapabilityPipelineError(
            "ontology and segregation manifest require artifact role v3"
        )

    validated_results: list[dict[str, Any]] = []
    for result in probe_results:
        validate_probe_result(result)
        if result["record_id"] not in record_ids:
            raise CapabilityPipelineError("probe result references a missing bundle record")
        if result["source_manifest_sha256"] not in source_hashes:
            raise CapabilityPipelineError("probe result references a missing bundle source")
        validated_results.append(dict(result))
    validated_results.sort(key=lambda row: row["probe_result_sha256"])
    if not validated_results:
        raise CapabilityPipelineError("bundle requires probe results")

    inventory_rows: list[dict[str, Any]] = []
    for inventory in inventories:
        validate_capability_inventory(inventory)
        if inventory["source_manifest_sha256"] not in source_hashes:
            raise CapabilityPipelineError("inventory references a missing bundle source")
        inventory_rows.append(dict(inventory))
    inventory_rows.sort(key=lambda row: row["inventory_sha256"])
    validate_user_selection_plan(selection)
    if not set(selection["selected_source_manifest_sha256"]).issubset(source_hashes):
        raise CapabilityPipelineError("selection references a missing bundle source")
    if artifact_role in training_roles:
        selected_item_keys = {
            (
                str(item["destination_scope"]),
                str(item["domain"]),
                str(item["capability"]),
                str(item["source_model"]),
                str(item["source_model_revision"]),
            )
            for item in selection["selected_items"]
        }
        record_item_keys = {
            (
                str(record["destination_scope"]),
                str(record["domain"]),
                str(record["capability"]),
                str(record["source_model"]),
                str(record["source_model_revision"]),
            )
            for record in validated_records
        }
        if record_item_keys != selected_item_keys:
            missing = sorted(selected_item_keys - record_item_keys)
            extra = sorted(record_item_keys - selected_item_keys)
            raise CapabilityPipelineError(
                "training records do not exactly match user selection; "
                f"missing={missing}, extra={extra}"
            )
    budget_rows: list[dict[str, Any]] = []
    prior_by_split: dict[str, set[str]] = {}
    for budget in budgets:
        _validate_budget_manifest(budget)
        ids = set(budget["record_ids"])
        if not ids.issubset(record_ids):
            raise CapabilityPipelineError("budget references a missing bundle record")
        previous = prior_by_split.setdefault(str(budget["split"]), set())
        if not previous.issubset(ids):
            raise CapabilityPipelineError("bundle budgets are not nested")
        prior_by_split[str(budget["split"])] = ids
        budget_rows.append(dict(budget))
    if not isinstance(ledger, Mapping) or not ledger:
        raise CapabilityPipelineError("bundle ledger is missing")

    payload_members = {
        "sources.json": canonical_json_bytes(validated_sources),
        "records.jsonl": _jsonl_bytes(validated_records),
        "probe_results.json": canonical_json_bytes(validated_results),
        "inventory.json": canonical_json_bytes(inventory_rows),
        "selection.json": canonical_json_bytes(selection),
        "budgets.json": canonical_json_bytes(budget_rows),
        "ledger.json": canonical_json_bytes(ledger),
    }
    if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE:
        payload_members.update(
            {
                "domain_ontology.json": canonical_json_bytes(
                    domain_ontology
                ),
                "segregation.json": canonical_json_bytes(
                    segregation_manifest
                ),
            }
        )
    manifest = {
        "schema_version": BUNDLE_MANIFEST_SCHEMA,
        "artifact_role": artifact_role,
        "installable_as_layercake_cake": False,
        "contains_teacher_material": True,
        "final_test_record_count": final_test_record_count,
        "final_test_records_allowed_for_training": False,
        "validation_records_allowed_for_training": False,
        "training_eligible": (
            artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
        ),
        "successor_promotion_eligible": (
            artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
            and selection["promotion_eligible"]
        ),
        "domain_segregation_required": (
            artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
        ),
        "absolute_zero_world_knowledge_claimed": False,
        "bounded_core_purity_manifest_sha256": (
            segregation_manifest["segregation_sha256"]
            if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
            else None
        ),
        "teacher_required_at_layercake_inference": False,
        "source_transformer_blocks_allowed_in_deployment": 0,
        "global_semantic_losslessness_claimed": False,
        "capability_scope": "probe_defined_not_exhaustive",
        "source_manifest_sha256": sorted(source_hashes),
        "selection_sha256": selection["selection_sha256"],
        "record_count": len(validated_records),
        "probe_result_count": len(validated_results),
        "inventory_count": len(inventory_rows),
        "budget_count": len(budget_rows),
        "members": {
            name: {"sha256": sha256_bytes(data), "bytes": len(data)}
            for name, data in sorted(payload_members.items())
        },
    }
    manifest["manifest_sha256"] = _mapping_hash(manifest, "manifest_sha256")
    members = dict(payload_members)
    members["manifest.json"] = canonical_json_bytes(manifest)
    raw = _zip_bytes(members)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(handle, "wb") as temporary:
            temporary.write(raw)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return {
        "path": str(output),
        "archive_sha256": sha256_bytes(raw),
        "archive_bytes": len(raw),
        "manifest_sha256": manifest["manifest_sha256"],
        "promotion_eligible_selection": selection["promotion_eligible"],
    }


def verify_extraction_bundle(
    path: str | Path, *, maximum_uncompressed_bytes: int = 2_000_000_000
) -> dict[str, Any]:
    """Verify structure, hashes, records, provenance, selection, and budgets."""

    raw = Path(path).read_bytes()
    archive_sha = sha256_bytes(raw)
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            name_set = frozenset(names)
            if len(names) != len(set(name.casefold() for name in names)):
                raise CapabilityPipelineError("duplicate or case-ambiguous bundle member")
            if name_set not in {
                _BUNDLE_MEMBERS,
                _SEGREGATED_BUNDLE_MEMBERS,
            }:
                raise CapabilityPipelineError("bundle member set is not schema-closed")
            if any(
                PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in PurePosixPath(name).parts)
                for name in names
            ):
                raise CapabilityPipelineError("unsafe bundle member path")
            if sum(info.file_size for info in infos) > maximum_uncompressed_bytes:
                raise CapabilityPipelineError("bundle exceeds uncompressed size limit")
            members = {name: archive.read(name) for name in names}
    except zipfile.BadZipFile as exc:
        raise CapabilityPipelineError("invalid extraction bundle ZIP") from exc

    manifest = json.loads(members["manifest.json"])
    _validate_self_hash(
        manifest,
        schema=BUNDLE_MANIFEST_SCHEMA,
        schema_field="schema_version",
        hash_field="manifest_sha256",
    )
    if manifest.get("installable_as_layercake_cake") is not False:
        raise CapabilityPipelineError("extraction bundle cannot be installable")
    if manifest.get("contains_teacher_material") is not True:
        raise CapabilityPipelineError("bundle must disclose teacher material")
    if manifest.get("global_semantic_losslessness_claimed") is not False:
        raise CapabilityPipelineError("bundle makes an invalid global losslessness claim")
    artifact_role = manifest.get("artifact_role")
    if artifact_role not in {
        SURVEY_ARTIFACT_ROLE,
        LEGACY_TRAINING_ARTIFACT_ROLE,
        TRAINING_ARTIFACT_ROLE,
        SEGREGATED_TRAINING_ARTIFACT_ROLE,
    }:
        raise CapabilityPipelineError("invalid extraction artifact role")
    expected_members = (
        _SEGREGATED_BUNDLE_MEMBERS
        if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
        else _BUNDLE_MEMBERS
    )
    if set(members) != expected_members:
        raise CapabilityPipelineError(
            "bundle member set does not match artifact role"
        )
    if manifest.get("final_test_records_allowed_for_training") is not False:
        raise CapabilityPipelineError("final-test training boundary is missing")
    if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE:
        if manifest.get("validation_records_allowed_for_training") is not False:
            raise CapabilityPipelineError("validation training boundary is missing")
        if manifest.get("training_eligible") is not True:
            raise CapabilityPipelineError("training eligibility is missing")
    member_index = manifest.get("members")
    if not isinstance(member_index, Mapping) or set(member_index) != (
        set(expected_members) - {"manifest.json"}
    ):
        raise CapabilityPipelineError(
            "manifest member index is not schema-exact"
        )
    for name, metadata in member_index.items():
        if name not in members or name == "manifest.json":
            raise CapabilityPipelineError("manifest member index is invalid")
        if metadata.get("bytes") != len(members[name]):
            raise CapabilityPipelineError(f"stale bundle member size: {name}")
        if metadata.get("sha256") != sha256_bytes(members[name]):
            raise CapabilityPipelineError(f"stale bundle member hash: {name}")

    sources = json.loads(members["sources.json"])
    for source in sources:
        validate_source_model_manifest(source)
    source_hashes = {source["source_manifest_sha256"] for source in sources}
    if manifest.get("source_manifest_sha256") != sorted(source_hashes):
        raise CapabilityPipelineError("source manifest index is stale")
    records = [
        json.loads(line)
        for line in members["records.jsonl"].splitlines()
        if line.strip()
    ]
    record_ids: set[str] = set()
    for record in records:
        validate_labeled_extraction_record(record)
        if record["record_id"] in record_ids:
            raise CapabilityPipelineError("duplicate record in bundle")
        record_ids.add(record["record_id"])
    actual_final_test_records = sum(
        1 for record in records if record["split"] == "final_test"
    )
    if manifest.get("final_test_record_count") != actual_final_test_records:
        raise CapabilityPipelineError("final-test record count is stale")
    if manifest.get("record_count") != len(records):
        raise CapabilityPipelineError("bundle record count is stale")
    if artifact_role == LEGACY_TRAINING_ARTIFACT_ROLE and actual_final_test_records:
        raise CapabilityPipelineError(
            "training artifact contains forbidden final-test records"
        )
    actual_non_search_records = sum(
        1 for record in records if record["split"] != "search"
    )
    if artifact_role in {
        TRAINING_ARTIFACT_ROLE,
        SEGREGATED_TRAINING_ARTIFACT_ROLE,
    } and actual_non_search_records:
        raise CapabilityPipelineError(
            "training artifact contains forbidden validation/final-test records"
        )
    if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE:
        if manifest.get("domain_segregation_required") is not True:
            raise CapabilityPipelineError(
                "segregated bundle does not require domain segregation"
            )
        if manifest.get("absolute_zero_world_knowledge_claimed") is not False:
            raise CapabilityPipelineError(
                "segregated bundle makes an absolute purity claim"
            )
        domain_ontology = json.loads(members["domain_ontology.json"])
        segregation_manifest = json.loads(members["segregation.json"])
        try:
            validate_core_domain_segregation_manifest(
                segregation_manifest,
                records,
                domain_ontology=domain_ontology,
            )
        except AcquisitionAccountingError as exc:
            raise CapabilityPipelineError(
                f"verified core/domain segregation gate failed: {exc}"
            ) from exc
        if (
            manifest.get("bounded_core_purity_manifest_sha256")
            != segregation_manifest["segregation_sha256"]
        ):
            raise CapabilityPipelineError(
                "bundle purity manifest identity is stale"
            )
    else:
        domain_ontology = None
        segregation_manifest = None
    results = json.loads(members["probe_results.json"])
    for result in results:
        validate_probe_result(result)
        if result["record_id"] not in record_ids:
            raise CapabilityPipelineError("probe result lacks its record")
        if result["source_manifest_sha256"] not in source_hashes:
            raise CapabilityPipelineError("probe result lacks its source")
    if manifest.get("probe_result_count") != len(results):
        raise CapabilityPipelineError("bundle probe-result count is stale")
    inventories = json.loads(members["inventory.json"])
    for inventory in inventories:
        validate_capability_inventory(inventory)
    if manifest.get("inventory_count") != len(inventories):
        raise CapabilityPipelineError("bundle inventory count is stale")
    selection = json.loads(members["selection.json"])
    validate_user_selection_plan(selection)
    if manifest.get("selection_sha256") != selection["selection_sha256"]:
        raise CapabilityPipelineError("bundle selection identity is stale")
    if artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE and (
        manifest.get("successor_promotion_eligible")
        is not selection["promotion_eligible"]
    ):
        raise CapabilityPipelineError(
            "successor promotion eligibility is stale"
        )
    if artifact_role in {
        TRAINING_ARTIFACT_ROLE,
        SEGREGATED_TRAINING_ARTIFACT_ROLE,
    }:
        selected_item_keys = {
            (
                str(item["destination_scope"]),
                str(item["domain"]),
                str(item["capability"]),
                str(item["source_model"]),
                str(item["source_model_revision"]),
            )
            for item in selection["selected_items"]
        }
        record_item_keys = {
            (
                str(record["destination_scope"]),
                str(record["domain"]),
                str(record["capability"]),
                str(record["source_model"]),
                str(record["source_model_revision"]),
            )
            for record in records
        }
        if record_item_keys != selected_item_keys:
            raise CapabilityPipelineError(
                "verified training records do not exactly match user selection"
            )
    budgets = json.loads(members["budgets.json"])
    if manifest.get("budget_count") != len(budgets):
        raise CapabilityPipelineError("bundle budget count is stale")
    prior_by_split: dict[str, set[str]] = {}
    for budget in budgets:
        _validate_budget_manifest(budget)
        ids = set(budget["record_ids"])
        if not ids.issubset(record_ids):
            raise CapabilityPipelineError("budget references unknown records")
        previous = prior_by_split.setdefault(str(budget["split"]), set())
        if not previous.issubset(ids):
            raise CapabilityPipelineError("verified budgets are not nested")
        prior_by_split[str(budget["split"])] = ids
    return {
        "verified": True,
        "archive_sha256": archive_sha,
        "archive_bytes": len(raw),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_count": len(sources),
        "record_count": len(records),
        "probe_result_count": len(results),
        "inventory_count": len(inventories),
        "budget_count": len(budgets),
        "promotion_eligible_selection": selection["promotion_eligible"],
        "artifact_role": artifact_role,
        "training_eligible": (
            artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
        ),
        "historical_manifest_training_eligible": manifest.get(
            "training_eligible"
        ),
        "successor_promotion_eligible": (
            artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
            and selection["promotion_eligible"]
        ),
        "domain_segregation_verified": (
            artifact_role == SEGREGATED_TRAINING_ARTIFACT_ROLE
        ),
        "selected_domains": (
            segregation_manifest["selected_domains"]
            if segregation_manifest is not None
            else []
        ),
    }


def read_extraction_bundle(path: str | Path) -> dict[str, Any]:
    """Return verified bundle contents for deterministic multi-source composition."""

    verification = verify_extraction_bundle(path)
    with zipfile.ZipFile(path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    return {
        "verification": verification,
        "manifest": json.loads(members["manifest.json"]),
        "sources": json.loads(members["sources.json"]),
        "records": [
            json.loads(line)
            for line in members["records.jsonl"].splitlines()
            if line.strip()
        ],
        "probe_results": json.loads(members["probe_results.json"]),
        "inventories": json.loads(members["inventory.json"]),
        "selection": json.loads(members["selection.json"]),
        "budgets": json.loads(members["budgets.json"]),
        "ledger": json.loads(members["ledger.json"]),
        "domain_ontology": (
            json.loads(members["domain_ontology.json"])
            if "domain_ontology.json" in members
            else None
        ),
        "segregation": (
            json.loads(members["segregation.json"])
            if "segregation.json" in members
            else None
        ),
    }


def build_semantic_retention_certificate(
    *,
    extraction_archive_sha256: str,
    deployed_artifact_sha256_before: str,
    deployed_artifact_sha256_after: str,
    evaluations: Sequence[Mapping[str, Any]],
    teacher_present_at_inference: bool,
    source_transformer_blocks_retained: int,
    minimum_distinct_prompts_per_destination: int = 100,
    minimum_seeds: int = 3,
) -> dict[str, Any]:
    """Certify exact payload identity and bounded, paired semantic retention.

    Each evaluation row must contain ``destination``, ``prompt_id``, ``seed``,
    ``source_passed``, ``layercake_passed``, and ``critical``.  Promotion
    requires no source-passing prompt to regress on LayerCake, all critical
    rows to pass, the locked prompt/seed depth, teacher absence, zero retained
    source blocks, and unchanged deployed artifact bytes.
    """

    _require_sha256("extraction_archive_sha256", extraction_archive_sha256)
    before = _require_sha256(
        "deployed_artifact_sha256_before", deployed_artifact_sha256_before
    )
    after = _require_sha256(
        "deployed_artifact_sha256_after", deployed_artifact_sha256_after
    )
    if not isinstance(teacher_present_at_inference, bool):
        raise CapabilityPipelineError("teacher_present_at_inference must be boolean")
    source_transformer_blocks_retained = _require_nonnegative_int(
        "source_transformer_blocks_retained", source_transformer_blocks_retained
    )
    minimum_distinct_prompts_per_destination = _require_nonnegative_int(
        "minimum_distinct_prompts_per_destination",
        minimum_distinct_prompts_per_destination,
    )
    minimum_seeds = _require_nonnegative_int("minimum_seeds", minimum_seeds)
    if minimum_distinct_prompts_per_destination < 1 or minimum_seeds < 1:
        raise CapabilityPipelineError("retention depth requirements must be positive")
    if not evaluations:
        raise CapabilityPipelineError("retention certificate requires evaluations")

    destinations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for row in evaluations:
        destination = _require_string("evaluation destination", row.get("destination"))
        prompt_id = _require_string("evaluation prompt_id", row.get("prompt_id"))
        seed = row.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CapabilityPipelineError("evaluation seed must be non-negative")
        for field in ("source_passed", "layercake_passed", "critical"):
            if not isinstance(row.get(field), bool):
                raise CapabilityPipelineError(f"evaluation {field} must be boolean")
        key = (destination, prompt_id, seed)
        if key in seen:
            raise CapabilityPipelineError(f"duplicate retention evaluation: {key}")
        seen.add(key)
        destinations[destination].append(row)

    summaries: list[dict[str, Any]] = []
    depth_pass = True
    no_regressions = True
    critical_pass = True
    for destination, rows in sorted(destinations.items()):
        prompts = {str(row["prompt_id"]) for row in rows}
        seeds = {int(row["seed"]) for row in rows}
        regressions = sum(
            1
            for row in rows
            if row["source_passed"] and not row["layercake_passed"]
        )
        critical_failures = sum(
            1 for row in rows if row["critical"] and not row["layercake_passed"]
        )
        source_successes = sum(1 for row in rows if row["source_passed"])
        layercake_successes = sum(1 for row in rows if row["layercake_passed"])
        destination_depth = (
            len(prompts) >= minimum_distinct_prompts_per_destination
            and len(seeds) >= minimum_seeds
        )
        depth_pass = depth_pass and destination_depth
        no_regressions = no_regressions and regressions == 0
        critical_pass = critical_pass and critical_failures == 0
        summaries.append(
            {
                "destination": destination,
                "observation_count": len(rows),
                "distinct_prompt_count": len(prompts),
                "seed_count": len(seeds),
                "source_successes": source_successes,
                "layercake_successes": layercake_successes,
                "source_to_layercake_regressions": regressions,
                "critical_failures": critical_failures,
                "depth_pass": destination_depth,
            }
        )

    payload_identity = before == after
    deployment_is_teacher_free = (
        teacher_present_at_inference is False
        and source_transformer_blocks_retained == 0
    )
    passed = (
        payload_identity
        and deployment_is_teacher_free
        and depth_pass
        and no_regressions
        and critical_pass
    )
    certificate = {
        "schema_version": RETENTION_CERTIFICATE_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "extraction_archive_sha256": extraction_archive_sha256,
        "deployed_artifact_sha256_before": before,
        "deployed_artifact_sha256_after": after,
        "payload_byte_identity": payload_identity,
        "teacher_present_at_inference": teacher_present_at_inference,
        "source_transformer_blocks_retained": source_transformer_blocks_retained,
        "teacher_free_deployment": deployment_is_teacher_free,
        "minimum_distinct_prompts_per_destination": (
            minimum_distinct_prompts_per_destination
        ),
        "minimum_seeds": minimum_seeds,
        "destination_summaries": summaries,
        "all_depth_gates_pass": depth_pass,
        "zero_measured_source_to_layercake_regressions": no_regressions,
        "all_critical_rows_pass": critical_pass,
        "semantic_claim_scope": (
            "zero_measured_regressions_on_locked_probe_suite"
            if passed
            else "not_certified"
        ),
        "global_semantic_identity_claimed": False,
    }
    certificate["certificate_sha256"] = _mapping_hash(
        certificate, "certificate_sha256"
    )
    return certificate
