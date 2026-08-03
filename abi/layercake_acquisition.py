"""Fail-closed accounting for ABI-to-LayerCake capability acquisition.

This module does not claim that ABI transfer works.  It makes every channel by
which information can cross from a source model explicit, labels each extracted
record for the English core or a domain cake, and selects only among
preregistered nested budgets.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


RECORD_SCHEMA = "abi-layercake-labeled-extraction-record/1"
LEDGER_SCHEMA = "abi-layercake-imported-information-ledger/1"
BUDGET_DECISION_SCHEMA = "abi-layercake-minimum-passing-budget-decision/1"

DESTINATION_SCOPES = frozenset({"english_core", "domain_cake"})
SPLITS = frozenset({"search", "validation", "final_test"})
ENGLISH_CORE_CAPABILITIES = frozenset(
    {
        "grammar",
        "coherence",
        "prompt_grounding",
        "instruction_following",
        "conversation",
        "summarization",
        "rewriting",
        "email_drafting",
        "tone_control",
        "format_control",
        "clarification",
        "abstention",
        "domain_independent_reasoning",
        "cake_output_realization",
    }
)


class AcquisitionAccountingError(ValueError):
    """Raised when acquisition evidence is incomplete or internally inconsistent."""


def _canonical_sha(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcquisitionAccountingError(f"{name} must be a non-empty string")
    return value


def _require_nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AcquisitionAccountingError(f"{name} must be a non-negative integer")
    return value


def _require_nonnegative_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise AcquisitionAccountingError(f"{name} must be non-negative")
    return float(value)


def build_labeled_extraction_record(
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
    authoritative_generated_token_ids: Sequence[int] | None = None,
    finish_reason: str | None = None,
    generation_max_new_tokens: int | None = None,
    teacher_input_tokens: int | None = None,
) -> dict[str, Any]:
    """Build one content-addressed source extraction record.

    ``teacher_tokens`` must come from the source runtime or its authoritative
    tokenizer.  Estimated token counts are not accepted.
    """

    if destination_scope not in DESTINATION_SCOPES:
        raise AcquisitionAccountingError("invalid destination_scope")
    capability = _require_nonempty("capability", capability)
    domain = _require_nonempty("domain", domain)
    if split not in SPLITS:
        raise AcquisitionAccountingError("invalid split")
    if destination_scope == "english_core":
        if domain != "domain_independent":
            raise AcquisitionAccountingError(
                "English-core extraction records must be domain_independent"
            )
        if capability not in ENGLISH_CORE_CAPABILITIES:
            raise AcquisitionAccountingError(
                f"{capability!r} is not a locked English-core capability"
            )
    elif domain == "domain_independent":
        raise AcquisitionAccountingError(
            "domain_cake records must name their specialist domain"
        )
    _require_nonempty("provenance", provenance)
    _require_nonempty("source_model", source_model)
    _require_nonempty("source_model_revision", source_model_revision)
    if not isinstance(prompt, str) or not isinstance(output, str):
        raise AcquisitionAccountingError("prompt and output must be strings")
    _require_nonempty("teacher_token_counter", teacher_token_counter)
    _require_nonnegative_int("teacher_tokens", teacher_tokens)

    runtime_values = (
        authoritative_generated_token_ids,
        finish_reason,
        generation_max_new_tokens,
        teacher_input_tokens,
    )
    if any(value is not None for value in runtime_values) and not all(
        value is not None for value in runtime_values
    ):
        raise AcquisitionAccountingError(
            "runtime generation evidence must be complete when present"
        )
    generated_ids: list[int] | None = None
    if authoritative_generated_token_ids is not None:
        if isinstance(authoritative_generated_token_ids, (str, bytes)):
            raise AcquisitionAccountingError(
                "authoritative generated token IDs must be an integer sequence"
            )
        generated_ids = list(authoritative_generated_token_ids)
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            for token_id in generated_ids
        ):
            raise AcquisitionAccountingError(
                "authoritative generated token IDs must be non-negative integers"
            )
        if len(generated_ids) != teacher_tokens:
            raise AcquisitionAccountingError(
                "teacher token count does not match authoritative generated IDs"
            )
        if teacher_token_counter != "authoritative_generated_token_ids":
            raise AcquisitionAccountingError(
                "retained token IDs require the authoritative runtime counter"
            )
        if finish_reason not in {"eos_token", "length"}:
            raise AcquisitionAccountingError("invalid source finish_reason")
        maximum = _require_nonnegative_int(
            "generation_max_new_tokens", generation_max_new_tokens
        )
        if maximum < 1 or teacher_tokens > maximum:
            raise AcquisitionAccountingError(
                "generated token count exceeds max_new_tokens"
            )
        if finish_reason == "length" and teacher_tokens != maximum:
            raise AcquisitionAccountingError(
                "length termination must reach max_new_tokens exactly"
            )
        _require_nonnegative_int("teacher_input_tokens", teacher_input_tokens)

    record = {
        "schema_version": RECORD_SCHEMA,
        "destination_scope": destination_scope,
        "capability": capability,
        "domain": domain,
        "provenance": provenance,
        "split": split,
        "source_model": source_model,
        "source_model_revision": source_model_revision,
        "prompt": prompt,
        "prompt_sha256": _text_sha(prompt),
        "prompt_utf8_bytes": len(prompt.encode("utf-8")),
        "output": output,
        "output_sha256": _text_sha(output),
        "output_utf8_bytes": len(output.encode("utf-8")),
        "teacher_tokens": teacher_tokens,
        "teacher_token_counter": teacher_token_counter,
        "teacher_token_count_authoritative": True,
    }
    if generated_ids is not None:
        record.update(
            {
                "authoritative_generated_token_ids": generated_ids,
                "finish_reason": finish_reason,
                "generation_max_new_tokens": generation_max_new_tokens,
                "teacher_input_tokens": teacher_input_tokens,
            }
        )
    record["record_id"] = _canonical_sha(record)
    return record


def validate_labeled_extraction_record(record: Mapping[str, Any]) -> None:
    """Recompute hashes and enforce the destination boundary for one record."""

    if record.get("schema_version") == (
        "abi-layercake-segregated-extraction-record/2"
    ):
        from .capability_segregation import (
            validate_segregated_extraction_record,
        )

        validate_segregated_extraction_record(record)
        return
    rebuilt = build_labeled_extraction_record(
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
        authoritative_generated_token_ids=record.get(
            "authoritative_generated_token_ids"
        ),
        finish_reason=record.get("finish_reason"),
        generation_max_new_tokens=record.get("generation_max_new_tokens"),
        teacher_input_tokens=record.get("teacher_input_tokens"),
    )
    for field in (
        "schema_version",
        "prompt_sha256",
        "prompt_utf8_bytes",
        "output_sha256",
        "output_utf8_bytes",
        "teacher_token_count_authoritative",
        "record_id",
    ):
        if record.get(field) != rebuilt[field]:
            raise AcquisitionAccountingError(f"stale or invalid record field: {field}")


def build_imported_information_ledger(
    records: Sequence[Mapping[str, Any]],
    *,
    logits_stored_count: int,
    logits_stored_bytes: int,
    hidden_activations_stored_count: int,
    hidden_activations_stored_bytes: int,
    frozen_source_parameters_copied: int,
    frozen_source_parameter_bytes_copied: int,
    final_imported_substrate_parameters: int,
    final_imported_substrate_parameter_bytes: int,
    bridge_parameters_trained: int,
    bridge_parameter_bytes: int,
    artifact_disk_footprint_bytes: int,
    peak_process_resident_memory_bytes: int,
    cpu_core_hours: float,
    source_model_inference_hours: float,
    one_time_source_extraction_seconds: float,
    per_host_acquisition_and_certification_seconds: float,
    final_deployed_footprint_bytes: int,
    final_cpu_inference_seconds: float,
    active_parameter_seconds: float,
    external_hardware_used: bool,
    external_hardware_description: str,
) -> dict[str, Any]:
    """Account every transfer channel without treating text as the only payload."""

    if not records:
        raise AcquisitionAccountingError("at least one extraction record is required")
    validated: list[Mapping[str, Any]] = []
    record_ids: set[str] = set()
    source_identities: set[tuple[str, str]] = set()
    for record in records:
        validate_labeled_extraction_record(record)
        record_id = str(record["record_id"])
        if record_id in record_ids:
            raise AcquisitionAccountingError(f"duplicate record_id: {record_id}")
        record_ids.add(record_id)
        source_identities.add(
            (str(record["source_model"]), str(record["source_model_revision"]))
        )
        validated.append(record)

    integer_inputs = {
        "logits_stored_count": logits_stored_count,
        "logits_stored_bytes": logits_stored_bytes,
        "hidden_activations_stored_count": hidden_activations_stored_count,
        "hidden_activations_stored_bytes": hidden_activations_stored_bytes,
        "frozen_source_parameters_copied": frozen_source_parameters_copied,
        "frozen_source_parameter_bytes_copied": frozen_source_parameter_bytes_copied,
        "final_imported_substrate_parameters": final_imported_substrate_parameters,
        "final_imported_substrate_parameter_bytes": (
            final_imported_substrate_parameter_bytes
        ),
        "bridge_parameters_trained": bridge_parameters_trained,
        "bridge_parameter_bytes": bridge_parameter_bytes,
        "artifact_disk_footprint_bytes": artifact_disk_footprint_bytes,
        "peak_process_resident_memory_bytes": peak_process_resident_memory_bytes,
        "final_deployed_footprint_bytes": final_deployed_footprint_bytes,
    }
    integers = {
        name: _require_nonnegative_int(name, value)
        for name, value in integer_inputs.items()
    }
    numeric_inputs = {
        "cpu_core_hours": cpu_core_hours,
        "source_model_inference_hours": source_model_inference_hours,
        "one_time_source_extraction_seconds": one_time_source_extraction_seconds,
        "per_host_acquisition_and_certification_seconds": (
            per_host_acquisition_and_certification_seconds
        ),
        "final_cpu_inference_seconds": final_cpu_inference_seconds,
        "active_parameter_seconds": active_parameter_seconds,
    }
    numbers = {
        name: _require_nonnegative_number(name, value)
        for name, value in numeric_inputs.items()
    }
    if not isinstance(external_hardware_used, bool):
        raise AcquisitionAccountingError("external_hardware_used must be boolean")
    if external_hardware_used:
        _require_nonempty(
            "external_hardware_description", external_hardware_description
        )
    elif external_hardware_description.strip():
        raise AcquisitionAccountingError(
            "external_hardware_description must be empty when no external hardware was used"
        )

    raw_prompt_bytes = sum(int(row["prompt_utf8_bytes"]) for row in validated)
    output_bytes = sum(int(row["output_utf8_bytes"]) for row in validated)
    teacher_tokens = sum(int(row["teacher_tokens"]) for row in validated)
    unique_payloads: dict[str, int] = {}
    unique_outputs: dict[str, tuple[int, int]] = {}
    for row in validated:
        unique_payloads.setdefault(
            str(row["prompt_sha256"]), int(row["prompt_utf8_bytes"])
        )
        unique_payloads.setdefault(
            str(row["output_sha256"]), int(row["output_utf8_bytes"])
        )
        unique_outputs.setdefault(
            str(row["output_sha256"]),
            (int(row["output_utf8_bytes"]), int(row["teacher_tokens"])),
        )
    unique_utf8_bytes = sum(unique_payloads.values())
    duplicate_adjusted_teacher_bytes = sum(value[0] for value in unique_outputs.values())
    duplicate_adjusted_teacher_tokens = sum(value[1] for value in unique_outputs.values())
    text_payload_bytes = raw_prompt_bytes + output_bytes
    total_accounted_transfer_bytes = (
        text_payload_bytes
        + integers["logits_stored_bytes"]
        + integers["hidden_activations_stored_bytes"]
        + integers["frozen_source_parameter_bytes_copied"]
        + integers["final_imported_substrate_parameter_bytes"]
        + integers["bridge_parameter_bytes"]
    )

    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "record_ids": sorted(record_ids),
        "source_identities": [
            {"model": model, "revision": revision}
            for model, revision in sorted(source_identities)
        ],
        "destination_scopes": sorted(
            {str(row["destination_scope"]) for row in validated}
        ),
        "splits": sorted({str(row["split"]) for row in validated}),
        "raw_source_prompt_count": len(validated),
        "raw_source_prompt_bytes": raw_prompt_bytes,
        "unique_utf8_bytes": unique_utf8_bytes,
        "teacher_generated_output_bytes": output_bytes,
        "teacher_tokens": teacher_tokens,
        "duplicate_adjusted_teacher_bytes": duplicate_adjusted_teacher_bytes,
        "duplicate_adjusted_teacher_tokens": duplicate_adjusted_teacher_tokens,
        **integers,
        **numbers,
        "external_hardware_used": external_hardware_used,
        "external_hardware_description": external_hardware_description,
        "total_accounted_transfer_bytes": total_accounted_transfer_bytes,
        "total_imported_payload_bits": total_accounted_transfer_bytes * 8,
        "logical_payload_rule": (
            "The total intentionally counts text, stored logits, stored hidden "
            "activations, copied source parameters, final imported substrate "
            "parameters, and trained bridge parameters as separate disclosed "
            "channels; overlaps are not hidden by compression."
        ),
    }
    ledger["ledger_sha256"] = _canonical_sha(ledger)
    return ledger


def validate_nested_budget_observations(
    observations: Sequence[Mapping[str, Any]],
) -> None:
    """Validate increasing, nested validation budgets and complete boolean gates."""

    if not observations:
        raise AcquisitionAccountingError("no budget observations")
    ordered = sorted(observations, key=lambda row: int(row["teacher_tokens"]))
    if list(observations) != ordered:
        raise AcquisitionAccountingError(
            "budget observations must be ordered by teacher_tokens"
        )
    prior_ids: set[str] = set()
    prior_tokens = -1
    for row in observations:
        if row.get("split") != "validation":
            raise AcquisitionAccountingError(
                "minimum-budget selection may use validation observations only"
            )
        tokens = _require_nonnegative_int("teacher_tokens", row.get("teacher_tokens"))
        _require_nonnegative_int(
            "total_imported_payload_bits", row.get("total_imported_payload_bits")
        )
        if tokens <= prior_tokens:
            raise AcquisitionAccountingError("teacher-token budgets must be unique")
        record_ids = row.get("record_ids")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or any(not isinstance(value, str) or not value for value in record_ids)
        ):
            raise AcquisitionAccountingError("every budget needs non-empty record_ids")
        current_ids = set(record_ids)
        if len(current_ids) != len(record_ids):
            raise AcquisitionAccountingError("record_ids within a budget must be unique")
        if not prior_ids.issubset(current_ids):
            raise AcquisitionAccountingError("larger budgets must nest smaller budgets")
        gates = row.get("common_gates")
        if not isinstance(gates, dict) or not gates:
            raise AcquisitionAccountingError("common_gates must be a non-empty object")
        if any(not isinstance(value, bool) for value in gates.values()):
            raise AcquisitionAccountingError("every common gate must be explicitly boolean")
        prior_ids = current_ids
        prior_tokens = tokens


def select_minimum_passing_budget(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the lowest tested passing teacher-token budget, then imported bits."""

    validate_nested_budget_observations(observations)
    passing = [
        row for row in observations if all(row["common_gates"].values())
    ]
    if not passing:
        raise AcquisitionAccountingError("no tested acquisition budget passes all gates")
    selected = min(
        passing,
        key=lambda row: (
            int(row["teacher_tokens"]),
            int(row["total_imported_payload_bits"]),
        ),
    )
    lower_failures = [
        row
        for row in observations
        if int(row["teacher_tokens"]) < int(selected["teacher_tokens"])
        and not all(row["common_gates"].values())
    ]
    largest_failing = max(
        lower_failures, key=lambda row: int(row["teacher_tokens"]), default=None
    )
    decision = {
        "schema_version": BUDGET_DECISION_SCHEMA,
        "claim": (
            "minimum passing budget among the preregistered tested nested budgets"
        ),
        "absolute_minimum_claimed": False,
        "selected_budget_id": selected.get("budget_id"),
        "selected_teacher_tokens": int(selected["teacher_tokens"]),
        "selected_total_imported_payload_bits": int(
            selected["total_imported_payload_bits"]
        ),
        "largest_lower_failing_budget_id": (
            largest_failing.get("budget_id") if largest_failing else None
        ),
        "largest_lower_failing_teacher_tokens": (
            int(largest_failing["teacher_tokens"]) if largest_failing else None
        ),
        "tested_budget_count": len(observations),
        "selection_split": "validation",
        "final_test_used_for_selection": False,
    }
    decision["decision_sha256"] = _canonical_sha(decision)
    return decision


def assert_deployed_layercake_is_teacher_free(
    manifest: Mapping[str, Any], *, expected_canonical_abi_sha256: str
) -> None:
    """Reject deployment manifests that retain a teacher or source transformer."""

    if manifest.get("teacher_present_at_inference") is not False:
        raise AcquisitionAccountingError("teacher absence must be explicit")
    if manifest.get("source_transformer_blocks_retained") not in (0, [], False):
        raise AcquisitionAccountingError("source transformer blocks remain deployed")
    if manifest.get("canonical_semantic_abi_sha256") != expected_canonical_abi_sha256:
        raise AcquisitionAccountingError("canonical LayerCake ABI changed")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise AcquisitionAccountingError("deployment components are missing")
    forbidden = {"source_teacher", "source_transformer_block", "retrieval_teacher"}
    if any(
        isinstance(component, Mapping) and component.get("type") in forbidden
        for component in components
    ):
        raise AcquisitionAccountingError("forbidden source component is deployed")


def record_ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return validated content IDs for constructing nested budget manifests."""

    values = []
    for record in records:
        validate_labeled_extraction_record(record)
        values.append(str(record["record_id"]))
    return values
