"""Fail-closed separation of English linguistic form from domain knowledge.

The checks in this module are deliberately bounded. They prevent mislabeled,
unaccounted, overlapping, or known-domain records from entering an English
core corpus. They cannot prove that a neural representation contains literally
zero world knowledge.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .layercake_acquisition import (
    AcquisitionAccountingError,
    build_labeled_extraction_record,
)


SEGREGATED_RECORD_SCHEMA = "abi-layercake-segregated-extraction-record/2"
DOMAIN_ONTOLOGY_SCHEMA = "abi-domain-ontology/1"
SEGREGATION_MANIFEST_SCHEMA = "abi-core-domain-segregation-manifest/1"

LINGUISTIC_FORM = "english_linguistic_form"
SPECIALIST_KNOWLEDGE = "specialist_domain_knowledge"
QUARANTINED = "ambiguous_or_cross_domain_quarantine"

ENGLISH_CONTENT_BASES = frozenset(
    {
        "abstract_or_nonce_content",
        "supplied_non_domain_context",
        "interpersonal_pragmatics",
        "domain_free_instruction",
    }
)
DOMAIN_CONTENT_BASES = frozenset(
    {
        "specialist_fact",
        "specialist_procedure",
        "specialist_reasoning",
        "specialist_code",
    }
)
QUARANTINE_CONTENT_BASES = frozenset(
    {
        "unclassified_content",
        "cross_domain_conflict",
        "label_review_required",
    }
)
LABEL_METHODS = frozenset(
    {
        "preregistered_catalog",
        "human_review",
        "classifier_then_human_review",
        "user_supplied_ontology",
    }
)


class CapabilitySegregationError(AcquisitionAccountingError):
    """Raised when English and domain acquisition material is not segregated."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CapabilitySegregationError(
            f"{name} must be a non-empty string"
        )
    return value


def _require_sha256(name: str, value: Any) -> str:
    value = _require_string(name, value)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CapabilitySegregationError(f"{name} must be lowercase SHA-256")
    return value


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _normalized_text_sha(value: str) -> str:
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


def build_segregated_extraction_record(
    *,
    destination_scope: str,
    capability: str,
    domain: str,
    provenance: str,
    split: str,
    source_model: str,
    source_model_revision: str,
    prompt: str,
    output: str,
    teacher_tokens: int,
    teacher_token_counter: str,
    knowledge_class: str,
    content_basis: str,
    domain_labels: Sequence[str],
    domain_claims: Sequence[str],
    label_method: str,
    label_evidence_sha256: str,
    output_introduces_unsupplied_facts: bool,
) -> dict[str, Any]:
    """Build a v2 record with an explicit knowledge-destination contract."""

    base = build_labeled_extraction_record(
        destination_scope=destination_scope,
        capability=capability,
        domain=domain,
        provenance=provenance,
        split=split,
        source_model=source_model,
        source_model_revision=source_model_revision,
        prompt=prompt,
        output=output,
        teacher_tokens=teacher_tokens,
        teacher_token_counter=teacher_token_counter,
    )
    if knowledge_class not in {
        LINGUISTIC_FORM,
        SPECIALIST_KNOWLEDGE,
        QUARANTINED,
    }:
        raise CapabilitySegregationError("invalid knowledge_class")
    labels = sorted(
        {_require_string("domain label", value) for value in domain_labels}
    )
    claims = sorted(
        {_require_string("domain claim", value) for value in domain_claims}
    )
    if len(labels) != len(domain_labels) or len(claims) != len(domain_claims):
        raise CapabilitySegregationError(
            "domain labels and claims must be unique and sorted"
        )
    if label_method not in LABEL_METHODS:
        raise CapabilitySegregationError("invalid label_method")
    _require_sha256("label_evidence_sha256", label_evidence_sha256)
    if not isinstance(output_introduces_unsupplied_facts, bool):
        raise CapabilitySegregationError(
            "output_introduces_unsupplied_facts must be boolean"
        )

    if destination_scope == "english_core":
        if knowledge_class != LINGUISTIC_FORM:
            raise CapabilitySegregationError(
                "English-core records must be linguistic form"
            )
        if content_basis not in ENGLISH_CONTENT_BASES:
            raise CapabilitySegregationError(
                "English-core content basis is not knowledge-minimized"
            )
        if labels or claims:
            raise CapabilitySegregationError(
                "English-core records cannot carry domain labels or claims"
            )
        if output_introduces_unsupplied_facts:
            raise CapabilitySegregationError(
                "English output introduces unsupplied facts"
            )
    elif destination_scope == "domain_cake":
        if knowledge_class == QUARANTINED:
            if domain != "quarantine":
                raise CapabilitySegregationError(
                    "quarantined material must use the quarantine destination"
                )
            if content_basis not in QUARANTINE_CONTENT_BASES:
                raise CapabilitySegregationError(
                    "invalid quarantine content basis"
                )
        elif knowledge_class != SPECIALIST_KNOWLEDGE:
            raise CapabilitySegregationError(
                "domain-cake records must be specialist knowledge"
            )
        else:
            if content_basis not in DOMAIN_CONTENT_BASES:
                raise CapabilitySegregationError(
                    "domain-cake content basis is not specialist"
                )
            if labels != [domain]:
                raise CapabilitySegregationError(
                    "a domain record must have exactly its destination label"
                )
            if not claims:
                raise CapabilitySegregationError(
                    "domain records require atomically labeled claims or skills"
                )

    record = dict(base)
    record.pop("record_id")
    record["schema_version"] = SEGREGATED_RECORD_SCHEMA
    record.update(
        {
            "knowledge_class": knowledge_class,
            "content_basis": content_basis,
            "domain_labels": labels,
            "domain_claims": claims,
            "label_method": label_method,
            "label_evidence_sha256": label_evidence_sha256,
            "output_introduces_unsupplied_facts": (
                output_introduces_unsupplied_facts
            ),
            "normalized_prompt_sha256": _normalized_text_sha(prompt),
            "normalized_output_sha256": _normalized_text_sha(output),
        }
    )
    record["record_id"] = _canonical_sha(record)
    return record


def validate_segregated_extraction_record(record: Mapping[str, Any]) -> None:
    """Recompute a v2 record and reject stale semantic labels."""

    rebuilt = build_segregated_extraction_record(
        destination_scope=record.get("destination_scope"),
        capability=record.get("capability"),
        domain=record.get("domain"),
        provenance=record.get("provenance"),
        split=record.get("split"),
        source_model=record.get("source_model"),
        source_model_revision=record.get("source_model_revision"),
        prompt=record.get("prompt"),
        output=record.get("output"),
        teacher_tokens=record.get("teacher_tokens"),
        teacher_token_counter=record.get("teacher_token_counter"),
        knowledge_class=record.get("knowledge_class"),
        content_basis=record.get("content_basis"),
        domain_labels=record.get("domain_labels", []),
        domain_claims=record.get("domain_claims", []),
        label_method=record.get("label_method"),
        label_evidence_sha256=record.get("label_evidence_sha256"),
        output_introduces_unsupplied_facts=record.get(
            "output_introduces_unsupplied_facts"
        ),
    )
    if dict(record) != rebuilt:
        raise CapabilitySegregationError(
            "segregated extraction record is stale or invalid"
        )


def build_domain_ontology(
    domains: Sequence[Mapping[str, Any]],
    *,
    ontology_id: str,
) -> dict[str, Any]:
    """Build a user-governed, non-exhaustive domain labeling ontology."""

    ontology_id = _require_string("ontology_id", ontology_id)
    if not domains:
        raise CapabilitySegregationError("domain ontology cannot be empty")
    rows: list[dict[str, Any]] = []
    domain_ids: set[str] = set()
    marker_owner: dict[str, str] = {}
    for domain in domains:
        domain_id = _require_string("domain_id", domain.get("domain_id"))
        if domain_id == "domain_independent" or domain_id in domain_ids:
            raise CapabilitySegregationError(
                "domain ontology identifiers must be unique specialist domains"
            )
        domain_ids.add(domain_id)
        capabilities = sorted(
            {
                _require_string("capability", value)
                for value in domain.get("capabilities", [])
            }
        )
        markers = sorted(
            {
                _normalized_text(
                    _require_string("core exclusion marker", value)
                )
                for value in domain.get("core_exclusion_markers", [])
            }
        )
        if not capabilities or not markers:
            raise CapabilitySegregationError(
                "each domain needs capabilities and core exclusion markers"
            )
        for marker in markers:
            owner = marker_owner.setdefault(marker, domain_id)
            if owner != domain_id:
                raise CapabilitySegregationError(
                    f"ambiguous core exclusion marker: {marker}"
                )
        rows.append(
            {
                "domain_id": domain_id,
                "description": _require_string(
                    "description", domain.get("description")
                ),
                "capabilities": capabilities,
                "core_exclusion_markers": markers,
                "label_evidence_sha256": _require_sha256(
                    "label_evidence_sha256",
                    domain.get("label_evidence_sha256"),
                ),
            }
        )
    ontology: dict[str, Any] = {
        "schema_version": DOMAIN_ONTOLOGY_SCHEMA,
        "ontology_id": ontology_id,
        "domains": sorted(rows, key=lambda row: row["domain_id"]),
        "discovery_exhaustive_claimed": False,
        "unlabeled_or_cross_domain_material_policy": "quarantine",
    }
    ontology["ontology_sha256"] = _canonical_sha(ontology)
    return ontology


def validate_domain_ontology(ontology: Mapping[str, Any]) -> None:
    """Rebuild the ontology and verify its identity."""

    rebuilt = build_domain_ontology(
        ontology.get("domains", []),
        ontology_id=ontology.get("ontology_id"),
    )
    if dict(ontology) != rebuilt:
        raise CapabilitySegregationError("domain ontology is stale or invalid")


def _marker_present(text: str, marker: str) -> bool:
    pattern = r"(?<!\w)" + re.escape(marker) + r"(?!\w)"
    return re.search(pattern, text) is not None


def build_core_domain_segregation_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    domain_ontology: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove bounded corpus separation before LayerCake training."""

    validate_domain_ontology(domain_ontology)
    if not records:
        raise CapabilitySegregationError(
            "segregation manifest requires extraction records"
        )
    ontology = {
        row["domain_id"]: row for row in domain_ontology["domains"]
    }
    core_ids: list[str] = []
    domain_ids: dict[str, list[str]] = {
        domain_id: [] for domain_id in sorted(ontology)
    }
    core_fingerprints: set[str] = set()
    domain_fingerprints: set[str] = set()
    record_ids: set[str] = set()
    teacher_tokens = {"english_core": 0, "domain_cake": 0}

    for record in records:
        validate_segregated_extraction_record(record)
        if record["knowledge_class"] == QUARANTINED:
            raise CapabilitySegregationError(
                "quarantined material cannot enter a training bundle"
            )
        record_id = str(record["record_id"])
        if record_id in record_ids:
            raise CapabilitySegregationError("duplicate record_id")
        record_ids.add(record_id)
        fingerprints = {
            str(record["normalized_prompt_sha256"]),
            str(record["normalized_output_sha256"]),
        }
        scope = str(record["destination_scope"])
        teacher_tokens[scope] += int(record["teacher_tokens"])
        if scope == "english_core":
            core_ids.append(record_id)
            core_fingerprints.update(fingerprints)
            combined = _normalized_text(
                f"{record['prompt']}\n{record['output']}"
            )
            hits = [
                f"{domain_id}:{marker}"
                for domain_id, domain in ontology.items()
                for marker in domain["core_exclusion_markers"]
                if _marker_present(combined, marker)
            ]
            if hits:
                raise CapabilitySegregationError(
                    "known domain marker entered English core: "
                    + ", ".join(sorted(hits))
                )
        else:
            domain_id = str(record["domain"])
            domain = ontology.get(domain_id)
            if domain is None:
                raise CapabilitySegregationError(
                    f"domain is absent from ontology: {domain_id}"
                )
            if record["capability"] not in domain["capabilities"]:
                raise CapabilitySegregationError(
                    "domain capability is absent from ontology"
                )
            domain_ids[domain_id].append(record_id)
            domain_fingerprints.update(fingerprints)

    overlap = core_fingerprints & domain_fingerprints
    if overlap:
        raise CapabilitySegregationError(
            "normalized English/domain content overlap detected"
        )
    if not core_ids:
        raise CapabilitySegregationError(
            "successor training requires an English core corpus"
        )
    selected_domains = {
        domain: sorted(ids)
        for domain, ids in domain_ids.items()
        if ids
    }
    gates = {
        "all_records_use_segregated_schema_v2": True,
        "english_records_are_linguistic_form_only": True,
        "english_domain_labels_empty": True,
        "english_domain_claims_empty": True,
        "english_outputs_introduce_no_declared_unsupplied_facts": True,
        "known_domain_marker_hits_in_english_zero": True,
        "normalized_core_domain_content_overlap_zero": True,
        "domain_records_have_exactly_one_destination_label": True,
        "domain_claims_atomically_labeled": True,
        "unknown_or_cross_domain_material_absent": True,
    }
    manifest: dict[str, Any] = {
        "schema_version": SEGREGATION_MANIFEST_SCHEMA,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "ontology_sha256": domain_ontology["ontology_sha256"],
        "record_ids": sorted(record_ids),
        "english_core_record_ids": sorted(core_ids),
        "domain_record_ids": selected_domains,
        "teacher_tokens_by_destination": teacher_tokens,
        "selected_domains": sorted(selected_domains),
        "unselected_domain_records_in_bundle": 0,
        "ambiguous_or_cross_domain_records_in_bundle": 0,
        "absolute_zero_world_knowledge_claimed": False,
        "bounded_purity_claim": (
            "The English training corpus contains only records declared and "
            "validated as linguistic form, has no domain labels or claims, "
            "contains none of the ontology's exclusion markers, and shares no "
            "normalized prompt/output payload with packaged domains."
        ),
        "gates": gates,
    }
    manifest["segregation_sha256"] = _canonical_sha(manifest)
    return manifest


def validate_core_domain_segregation_manifest(
    manifest: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    domain_ontology: Mapping[str, Any],
) -> None:
    """Rebuild a segregation manifest and compare every field."""

    rebuilt = build_core_domain_segregation_manifest(
        records, domain_ontology=domain_ontology
    )
    if dict(manifest) != rebuilt:
        raise CapabilitySegregationError(
            "core/domain segregation manifest is stale or invalid"
        )
