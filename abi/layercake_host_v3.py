"""Versioned LayerCake English acquisition consumer for segregated ABI bundles.

The frozen v47 consumer remains byte-identical for historical verification.
This successor validates the v3 segregation contract, materializes only
linguistic-form rows, and reuses the certified training/evaluation
implementation through isolated function-global rebinding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    passed = {
        str(result["record_id"]): bool(result["passed"])
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
        if passed.get(str(record["record_id"])) is not True:
            raise LayerCakeHostError(
                "failed source response crossed the training boundary"
            )
        capability = str(record["capability"])
        rows.append(
            {
                "record_id": str(record["record_id"]),
                "capability": capability,
                "route": legacy.route_for_capability(capability),
                "prompt": legacy.strip_source_chat_template(
                    str(record["prompt"])
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
    missing = sorted(
        set(legacy.CAPABILITY_TO_ROUTE)
        - {row["capability"] for row in rows}
    )
    if missing:
        raise LayerCakeHostError(
            f"selected budget lacks complete English capability coverage: {missing}"
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
