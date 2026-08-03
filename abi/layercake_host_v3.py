"""Versioned LayerCake English acquisition consumer for segregated ABI bundles.

The frozen v47 consumer remains byte-identical for historical verification.
This successor validates the v3 segregation contract, materializes only
linguistic-form rows, and reuses the certified training/evaluation
implementation through isolated function-global rebinding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
from pathlib import Path
from types import FunctionType
from typing import Any

from . import layercake_host as legacy
from .capability_pipeline import (
    SEGREGATED_TRAINING_ARTIFACT_ROLE,
    read_extraction_bundle,
)
from .capability_segregation import LINGUISTIC_FORM


LayerCakeHostError = legacy.LayerCakeHostError


def _materialize_training_prompt(
    *,
    record: Mapping[str, Any],
    probe_result: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> str:
    """Strip generated chat prompts or verify an exact raw contrastive row."""

    prompt = str(record["prompt"])
    counter = record.get("teacher_token_counter")
    if counter != (
        "authoritative_source_tokenizer_posthoc_on_contrastive_selection"
    ):
        return legacy.strip_source_chat_template(prompt)
    evaluator = probe_result.get("evaluator")
    qualification = ledger.get("contrastive_qualification")
    if not isinstance(evaluator, Mapping) or not isinstance(
        qualification, Mapping
    ):
        raise LayerCakeHostError(
            "raw contrastive prompt lacks qualification evidence"
        )
    evidence_hash = str(evaluator.get("contrastive_evidence_sha256", ""))
    observation_hash = str(
        evaluator.get("contrastive_observation_sha256", "")
    )
    provenance = f"contrastive:{evidence_hash}:{observation_hash}"
    if (
        evaluator.get("kind") != "counterbalanced_source_preference"
        or evaluator.get("teacher_generated_output") is not False
        or evaluator.get("prompt_contract_sha256")
        != hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        or evaluator.get("selected_output_sha256")
        != record.get("output_sha256")
        or float(evaluator.get("ab_margin", 0.0)) <= 0.0
        or float(evaluator.get("ba_margin", 0.0)) <= 0.0
        or qualification.get("evidence_sha256") != evidence_hash
        or record.get("provenance") != provenance
        or evaluator.get("source_manifest_sha256")
        not in set(ledger.get("source_manifest_sha256", []))
        or "authoritative_generated_token_ids" in record
        or "finish_reason" in record
    ):
        raise LayerCakeHostError(
            "raw contrastive prompt evidence is incomplete or stale"
        )
    return prompt


def _require_segregated_training_bundle(
    bundle: Mapping[str, Any],
) -> None:
    verification = bundle["verification"]
    if (
        verification["artifact_role"]
        != SEGREGATED_TRAINING_ARTIFACT_ROLE
        or verification["training_eligible"] is not True
        or verification["domain_segregation_verified"] is not True
    ):
        raise LayerCakeHostError(
            "bundle is not current segregated training material"
        )
    segregation = bundle.get("segregation")
    if (
        not isinstance(segregation, Mapping)
        or segregation.get("status") != "PASS"
        or segregation.get("absolute_zero_world_knowledge_claimed") is not False
    ):
        raise LayerCakeHostError(
            "bundle lacks a bounded passing segregation manifest"
        )


def load_english_training_rows(
    bundle_path: str | Path,
    *,
    budget_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load passing search rows after rechecking the English-only boundary."""

    bundle = read_extraction_bundle(bundle_path)
    _require_segregated_training_bundle(bundle)
    budgets = bundle["budgets"]
    if budget_index < 0:
        budget_index += len(budgets)
    if budget_index < 0 or budget_index >= len(budgets):
        raise LayerCakeHostError(
            "budget_index is outside the bundle budget list"
        )
    budget = budgets[budget_index]
    if budget["split"] != "search":
        raise LayerCakeHostError("host acquisition may use search budgets only")
    allowed = set(budget["record_ids"])
    results_by_record = {
        str(result["record_id"]): result
        for result in bundle["probe_results"]
    }
    rows: list[dict[str, Any]] = []
    for record in bundle["records"]:
        if record["record_id"] not in allowed:
            continue
        if record["destination_scope"] != "english_core":
            continue
        if (
            record.get("knowledge_class") != LINGUISTIC_FORM
            or record.get("domain_labels") != []
            or record.get("domain_claims") != []
            or record.get("output_introduces_unsupplied_facts") is not False
        ):
            raise LayerCakeHostError(
                "non-linguistic material crossed the English training boundary"
            )
        if record["split"] != "search":
            raise LayerCakeHostError(
                "non-search record crossed the training boundary"
            )
        result = results_by_record.get(str(record["record_id"]))
        if result is None or result.get("passed") is not True:
            raise LayerCakeHostError(
                "failed source response crossed the training boundary"
            )
        capability = str(record["capability"])
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "capability": capability,
                "route": legacy.route_for_capability(capability),
                "prompt": _materialize_training_prompt(
                    record=record,
                    probe_result=result,
                    ledger=bundle["ledger"],
                ),
                "response": str(record["output"]),
                "teacher_tokens": int(record["teacher_tokens"]),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(
                    record["source_model_revision"]
                ),
                "provenance": str(record["provenance"]),
            }
        )
    if not rows:
        raise LayerCakeHostError(
            "selected budget contains no English records"
        )
    requested = bundle["selection"].get("requested_english_capabilities")
    required_capabilities = (
        set(legacy.CAPABILITY_TO_ROUTE)
        if requested is None
        else {str(capability) for capability in requested}
    )
    if (
        not required_capabilities
        or not required_capabilities.issubset(legacy.CAPABILITY_TO_ROUTE)
    ):
        raise LayerCakeHostError(
            "training bundle English capability contract is invalid"
        )
    observed_capabilities = {row["capability"] for row in rows}
    missing = sorted(required_capabilities - observed_capabilities)
    if missing:
        raise LayerCakeHostError(
            f"selected budget lacks complete English capability coverage: {missing}"
        )
    unexpected = sorted(observed_capabilities - required_capabilities)
    if unexpected:
        raise LayerCakeHostError(
            f"selected budget crossed English capability scope: {unexpected}"
        )
    rows.sort(key=lambda row: row["record_id"])
    return rows, budget, bundle


def build_validation_rows(
    *,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    catalog_paths: Sequence[str | Path],
    capabilities: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Require a v3 training identity before delegating held-out row binding."""

    training = read_extraction_bundle(training_bundle_path)
    _require_segregated_training_bundle(training)
    return legacy.build_validation_rows(
        training_bundle_path=training_bundle_path,
        validation_bundle_paths=validation_bundle_paths,
        catalog_paths=catalog_paths,
        capabilities=capabilities,
    )


def _rebind(
    function: Callable[..., Any],
    **global_overrides: Any,
) -> Callable[..., Any]:
    """Clone a function with isolated globals; never mutate the frozen module."""

    namespace = dict(function.__globals__)
    namespace.update(global_overrides)
    rebound = FunctionType(
        function.__code__,
        namespace,
        name=function.__name__,
        argdefs=function.__defaults__,
        closure=function.__closure__,
    )
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__annotations__ = function.__annotations__
    rebound.__doc__ = function.__doc__
    return rebound


train_host_delta = _rebind(
    legacy.train_host_delta,
    load_english_training_rows=load_english_training_rows,
)
derive_symbolic_surface_host = _rebind(
    legacy.derive_symbolic_surface_host,
    load_english_training_rows=load_english_training_rows,
)
evaluate_host_semantics = _rebind(
    legacy.evaluate_host_semantics,
    build_validation_rows=build_validation_rows,
)
load_host_model = legacy.load_host_model

main = _rebind(
    legacy.main,
    train_host_delta=train_host_delta,
    derive_symbolic_surface_host=derive_symbolic_surface_host,
    evaluate_host_semantics=evaluate_host_semantics,
)


if __name__ == "__main__":
    raise SystemExit(main())
