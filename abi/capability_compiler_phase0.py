"""Fail-closed verifier for the ABI capability-compiler Phase 0 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping


REQUIRED_SYSTEMS = {"T0", "L0", "L1", "D0", "D1", "D2", "A0", "A1", "A2", "A3", "A4"}
REQUIRED_FAIRNESS_VIEWS = {
    "equal_imported_information",
    "equal_final_deployment_constraint",
    "matched_quality_frontier",
}
REQUIRED_ENGLISH_CAPABILITIES = {
    "grammar",
    "coherence",
    "prompt_grounding",
    "instruction_following",
    "conversation",
    "supplied_text_summarization",
    "rewriting",
    "email_drafting_from_notes",
    "tone_control",
    "format_control",
    "clarification",
    "abstention",
    "fact_free_reasoning",
    "fluent_realization",
}


def _mapping(value: object, name: str, errors: list[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def validate_protocol(protocol: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if protocol.get("format") != "abi-capability-compiler-phase0-protocol/1":
        errors.append("unexpected format")
    if protocol.get("status") != "PREREGISTERED_BEFORE_PHASE1_ARTIFACT_CONSTRUCTION_OR_NEW_TRAINING":
        errors.append("protocol is not preregistered before Phase 1 and training")
    if protocol.get("historical_evidence_changed") is not False:
        errors.append("historical evidence must remain unchanged")

    scope = _mapping(protocol.get("initial_campaign_scope"), "initial_campaign_scope", errors)
    if scope.get("teacher_count") != 1:
        errors.append("initial campaign must use exactly one pinned teacher")
    if scope.get("generalization_to_other_teachers_claimed") is not False:
        errors.append("initial campaign cannot claim other-teacher generalization")
    if scope.get("final_teacher_absent_at_layercake_inference") is not True:
        errors.append("teacher absence at LayerCake inference is mandatory")
    if scope.get("final_source_blocks_at_layercake_inference") != 0:
        errors.append("source blocks at LayerCake inference must be zero")

    ontology = _mapping(protocol.get("capability_ontology"), "capability_ontology", errors)
    if set(ontology.get("english_core", [])) != REQUIRED_ENGLISH_CAPABILITIES:
        errors.append("English capability ontology is incomplete or changed")
    if not str(ontology.get("fact_free_reasoning_rule", "")).strip():
        errors.append("fact-free reasoning boundary is missing")

    boundaries = _mapping(protocol.get("data_boundaries"), "data_boundaries", errors)
    fractions = sum(
        float(boundaries.get(key, -1))
        for key in (
            "search_and_acquisition_fraction",
            "development_fraction",
            "final_fraction",
        )
    )
    if abs(fractions - 1.0) > 1e-12:
        errors.append("data split fractions must sum to one")
    if boundaries.get("final_data_may_select") != []:
        errors.append("final data cannot select anything")
    for key in (
        "minimum_passing_acquisition_records_per_english_capability",
        "minimum_development_prompts_per_english_capability",
        "minimum_final_prompts_per_english_capability",
        "minimum_isolation_prompts_per_declared_domain",
        "minimum_adversarial_prompts_per_family",
    ):
        if int(boundaries.get(key, 0)) < 100:
            errors.append(f"{key} must be at least 100")

    systems = _mapping(protocol.get("mandatory_systems"), "mandatory_systems", errors)
    if set(systems) != REQUIRED_SYSTEMS:
        errors.append("mandatory comparison system set is incomplete or changed")
    for baseline in ("L0", "L1", "D0", "D1", "D2"):
        spec = _mapping(systems.get(baseline), baseline, errors)
        configurations = int(spec.get("maximum_development_configurations", 0))
        if configurations < 1 or configurations > 8:
            errors.append(f"{baseline} development configurations must be within 1..8")
    if _mapping(systems.get("L0"), "L0", errors).get("source_base_present_at_inference") is not True:
        errors.append("L0 must count its source base at inference")
    if _mapping(systems.get("L1"), "L1", errors).get("source_base_present_at_inference") is not True:
        errors.append("L1 must count its source base at inference")
    for baseline in ("D0", "D1", "D2", "A0"):
        if _mapping(systems.get(baseline), baseline, errors).get("teacher_absent_at_inference") is not True:
            errors.append(f"{baseline} must remove the teacher at inference")

    fairness = _mapping(protocol.get("fairness_views"), "fairness_views", errors)
    if set(fairness) != REQUIRED_FAIRNESS_VIEWS:
        errors.append("all three fairness views are mandatory")

    statistics = _mapping(protocol.get("statistics"), "statistics", errors)
    if int(statistics.get("paired_bootstrap_resamples", 0)) < 10000:
        errors.append("paired bootstrap requires at least 10,000 resamples")
    if int(statistics.get("headline_training_seeds", 0)) < 3:
        errors.append("headline training requires at least three seeds")
    if int(statistics.get("headline_quality_distinct_prompts_minimum", 0)) < 100:
        errors.append("headline quality requires at least 100 distinct prompts")
    if int(statistics.get("headline_runtime_repetitions_minimum", 0)) < 20:
        errors.append("headline runtime requires at least 20 repetitions")
    if int(statistics.get("p95_observations_minimum", 0)) < 100:
        errors.append("p95 requires at least 100 observations")
    if int(statistics.get("p99_observations_minimum", 0)) < 1000:
        errors.append("p99 requires at least 1,000 observations")

    quality = _mapping(protocol.get("common_quality_gates"), "common_quality_gates", errors)
    if float(quality.get("teacher_relative_paired_difference_lower_bound_minimum", -1)) < -0.05:
        errors.append("teacher-relative quality margin cannot be weaker than -0.05")
    if quality.get("repetition_collapse_count_maximum") != 0:
        errors.append("repetition collapse gate must be zero")

    segregation = _mapping(
        protocol.get("segregation_and_exclusion_gates"),
        "segregation_and_exclusion_gates",
        errors,
    )
    for key in (
        "specialist_records_in_english_artifact_maximum",
        "cross_destination_duplicate_clusters_maximum",
        "inactive_capability_execution_events_maximum",
        "core_bytes_changed_by_domain_install_remove_maximum",
    ):
        if segregation.get(key) != 0:
            errors.append(f"{key} must be zero")

    integrated = _mapping(protocol.get("integrated_layercake_gates"), "integrated_layercake_gates", errors)
    if float(integrated.get("cpu_throughput_ratio_vs_optimized_transformer_minimum", 0)) < 2.0:
        errors.append("LayerCake CPU throughput ratio gate must be at least 2x")
    if integrated.get("extra_load_probe_allowed") is not False:
        errors.append("cold timing cannot use an extra load probe")
    if integrated.get("same_final_checkpoint_for_all_evidence") is not True:
        errors.append("all evidence must use one final checkpoint")

    superiority = _mapping(protocol.get("product_superiority_gates"), "product_superiority_gates", errors)
    if superiority.get("universal_superiority_claim_allowed") is not False:
        errors.append("universal superiority claim must be prohibited")
    if int(superiority.get("versus_distillation_strict_efficiency_endpoints_required", 0)) < 2:
        errors.append("ABI must beat distillation on at least two strict efficiency endpoints")

    accounting = protocol.get("information_accounting")
    if not isinstance(accounting, list) or len(accounting) < 20:
        errors.append("information accounting is incomplete")
    stop_rules = protocol.get("stop_rules")
    if not isinstance(stop_rules, list) or len(stop_rules) < 8:
        errors.append("stop rules are incomplete")
    exit_requirements = _mapping(protocol.get("phase0_exit_requirements"), "phase0_exit_requirements", errors)
    if exit_requirements.get("new_training_performed") is not False:
        errors.append("Phase 0 must perform no new training")
    if any(value is None for value in _walk_values(protocol)):
        errors.append("protocol cannot contain null values")
    return errors


def _walk_values(value: object):
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)
    else:
        yield value


def load_protocol(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("protocol root must be an object")
    return value


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("ABI_CAPABILITY_COMPILER_PHASE0_PROTOCOL_V1.json"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol)
    errors = validate_protocol(protocol)
    result = {
        "format": "abi-capability-compiler-phase0-verification/1",
        "protocol": args.protocol.as_posix(),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
