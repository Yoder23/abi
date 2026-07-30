"""Versioned LayerCake domain acquisition consumer for segregated ABI bundles.

The historical v47 domain consumer remains byte-identical. This successor
requires v3 segregation evidence and exact specialist labels before reusing
the certified domain training and evaluation implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import FunctionType
from typing import Any

from . import layercake_domains as legacy
from .capability_pipeline import (
    SEGREGATED_TRAINING_ARTIFACT_ROLE,
    read_extraction_bundle,
)
from .capability_segregation import SPECIALIST_KNOWLEDGE
from .layercake_host import strip_source_chat_template


DomainConformanceError = legacy.DomainConformanceError
SUPPORTED_DOMAINS = legacy.SUPPORTED_DOMAINS


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
        raise DomainConformanceError(
            "bundle is not current segregated training material"
        )
    segregation = bundle.get("segregation")
    if (
        not isinstance(segregation, Mapping)
        or segregation.get("status") != "PASS"
        or segregation.get("absolute_zero_world_knowledge_claimed") is not False
    ):
        raise DomainConformanceError(
            "bundle lacks a bounded passing segregation manifest"
        )


def load_domain_training_rows(
    bundle_path: str | Path,
    *,
    domain: str,
    budget_index: int = -1,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Load one selected domain after rechecking every semantic label."""

    if domain not in SUPPORTED_DOMAINS:
        raise DomainConformanceError(f"unsupported domain: {domain}")
    bundle = read_extraction_bundle(bundle_path)
    _require_segregated_training_bundle(bundle)
    budgets = bundle["budgets"]
    if budget_index < 0:
        budget_index += len(budgets)
    if not 0 <= budget_index < len(budgets):
        raise DomainConformanceError("budget index is outside the bundle")
    budget = budgets[budget_index]
    if budget["split"] != "search":
        raise DomainConformanceError("domain training may use search only")
    selected = legacy._selected_domain_item(bundle, domain)
    allowed = set(budget["record_ids"])
    results = {
        str(result["record_id"]): result
        for result in bundle["probe_results"]
    }
    rows: list[dict[str, Any]] = []
    for record in bundle["records"]:
        if record["record_id"] not in allowed:
            continue
        if (
            record["destination_scope"] != "domain_cake"
            or record["domain"] != domain
        ):
            continue
        if (
            record.get("knowledge_class") != SPECIALIST_KNOWLEDGE
            or record.get("domain_labels") != [domain]
            or not record.get("domain_claims")
        ):
            raise DomainConformanceError(
                "unlabeled specialist material crossed the domain boundary"
            )
        if record["split"] != "search":
            raise DomainConformanceError(
                "non-search row crossed training boundary"
            )
        if (
            record["source_model"] != selected["source_model"]
            or record["source_model_revision"]
            != selected["source_model_revision"]
        ):
            raise DomainConformanceError(
                "unselected source crossed domain boundary"
            )
        result = results.get(str(record["record_id"]))
        if result is None or result["passed"] is not True:
            raise DomainConformanceError(
                "failed or unscored source row crossed training boundary"
            )
        prompt = strip_source_chat_template(str(record["prompt"]))
        response = str(record["output"])
        rows.append(
            {
                "id": str(record["record_id"]),
                "domain_id": domain,
                "capability": str(record["capability"]),
                "prompt": prompt,
                "response": response,
                "copy_lexemes": legacy._candidate_copy_lexemes(
                    prompt,
                    response,
                    result["evaluator"],
                ),
                "evaluator": dict(result["evaluator"]),
                "teacher_tokens": int(record["teacher_tokens"]),
                "prompt_utf8_bytes": int(record["prompt_utf8_bytes"]),
                "output_utf8_bytes": int(record["output_utf8_bytes"]),
                "source_model": str(record["source_model"]),
                "source_model_revision": str(
                    record["source_model_revision"]
                ),
                "provenance": str(record["provenance"]),
            }
        )
    rows.sort(key=lambda row: row["id"])
    if not rows:
        raise DomainConformanceError(
            f"budget contains no rows for {domain}"
        )
    return rows, budget, bundle


def build_domain_validation_rows(
    *,
    training_bundle_path: str | Path,
    validation_bundle_paths: Sequence[str | Path],
    domain: str,
) -> list[dict[str, Any]]:
    """Require a v3 training identity before binding validation evidence."""

    training = read_extraction_bundle(training_bundle_path)
    _require_segregated_training_bundle(training)
    return legacy.build_domain_validation_rows(
        training_bundle_path=training_bundle_path,
        validation_bundle_paths=validation_bundle_paths,
        domain=domain,
    )


def _rebind(
    function: Callable[..., Any],
    **global_overrides: Any,
) -> Callable[..., Any]:
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


train_domain_candidate = _rebind(
    legacy.train_domain_candidate,
    load_domain_training_rows=load_domain_training_rows,
)
evaluate_domain_candidate = _rebind(
    legacy.evaluate_domain_candidate,
    build_domain_validation_rows=build_domain_validation_rows,
)
package_validated_candidate = legacy.package_validated_candidate

main = _rebind(
    legacy.main,
    train_domain_candidate=train_domain_candidate,
    evaluate_domain_candidate=evaluate_domain_candidate,
    package_validated_candidate=package_validated_candidate,
)


if __name__ == "__main__":
    raise SystemExit(main())
