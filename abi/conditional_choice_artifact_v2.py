"""Build fail-closed ABI artifacts from finite source-weight choices.

Unlike sequence distillation, this pathway never asks the source to generate a
response. A frozen source scores a preregistered finite set, an independent
evaluator identifies correct selections, and only verified selections become
training records. The descriptor binds every input hash and must disclose when
the artifact is a selective subset of a failed broad source screen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from .capability_pipeline import (
    SEGREGATED_TRAINING_ARTIFACT_ROLE,
    build_capability_inventory,
    build_extraction_bundle,
    build_nested_teacher_budgets,
    build_probe_result,
    build_user_selection_plan,
    read_extraction_bundle,
    verify_extraction_bundle,
)
from .capability_segregation import (
    build_core_domain_segregation_manifest,
    build_segregated_extraction_record,
    validate_domain_ontology,
)
from .contrastive_source_artifact import _load_source_token_counter
from .hf_extraction import load_probe_catalog


DESCRIPTOR_SCHEMA = "abi-conditional-choice-artifact-descriptor/2"
COUNTER = "authoritative_source_tokenizer_posthoc_on_conditional_choice_selection"


class ConditionalChoiceArtifactV2Error(RuntimeError):
    """Raised when conditional-choice evidence cannot be reproduced exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _evidence_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value) + b"\n").hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConditionalChoiceArtifactV2Error(f"{name} must be a positive integer")
    return value


def validate_descriptor(descriptor: Mapping[str, Any]) -> None:
    """Validate the self-hashed dependency and selectivity contract."""

    if descriptor.get("schema_version") != DESCRIPTOR_SCHEMA:
        raise ConditionalChoiceArtifactV2Error("unsupported descriptor schema")
    expected = descriptor.get("descriptor_sha256")
    payload = {key: value for key, value in descriptor.items() if key != "descriptor_sha256"}
    if expected != _canonical_sha(payload):
        raise ConditionalChoiceArtifactV2Error("descriptor self-hash is stale")
    for field in (
        "catalog_sha256",
        "result_file_sha256",
        "raw_scores_sha256",
        "result_evidence_sha256",
        "source_archive_sha256",
        "source_manifest_sha256",
    ):
        value = descriptor.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ConditionalChoiceArtifactV2Error(f"invalid descriptor hash: {field}")
    _positive_integer("source_rows", descriptor.get("source_rows"))
    minimum = _positive_integer("minimum_packaged_rows", descriptor.get("minimum_packaged_rows"))
    if minimum > descriptor["source_rows"]:
        raise ConditionalChoiceArtifactV2Error("minimum packaged rows exceeds source rows")
    family_minimum = _positive_integer("minimum_packaged_rows_per_family", descriptor.get("minimum_packaged_rows_per_family"))
    if not isinstance(descriptor.get("campaign"), str) or not descriptor["campaign"]:
        raise ConditionalChoiceArtifactV2Error("descriptor campaign is missing")
    if not isinstance(descriptor.get("result_format"), str) or not descriptor["result_format"]:
        raise ConditionalChoiceArtifactV2Error("descriptor result format is missing")
    if not isinstance(descriptor.get("accepted_result_verdicts"), list) or not descriptor["accepted_result_verdicts"]:
        raise ConditionalChoiceArtifactV2Error("descriptor accepted verdicts are missing")
    if not isinstance(descriptor.get("allow_selective_packaging"), bool):
        raise ConditionalChoiceArtifactV2Error("descriptor selectivity flag is invalid")
    if family_minimum * 7 > descriptor["source_rows"]:
        raise ConditionalChoiceArtifactV2Error("family minimum is impossible")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConditionalChoiceArtifactV2Error(f"invalid JSONL line {line_number}") from exc
            if not isinstance(value, dict):
                raise ConditionalChoiceArtifactV2Error("raw score row is not an object")
            rows.append(value)
    return rows


def _validate_dependencies(
    *,
    descriptor: Mapping[str, Any],
    source_bundle_path: Path,
    catalog_path: Path,
    result_path: Path,
    raw_scores_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    for path in (source_bundle_path, catalog_path, result_path, raw_scores_path):
        if not path.is_file():
            raise ConditionalChoiceArtifactV2Error(f"required input is absent: {path}")
    if _file_sha(catalog_path) != descriptor["catalog_sha256"]:
        raise ConditionalChoiceArtifactV2Error("catalog hash changed")
    if _file_sha(result_path) != descriptor["result_file_sha256"]:
        raise ConditionalChoiceArtifactV2Error("result file hash changed")
    if _file_sha(raw_scores_path) != descriptor["raw_scores_sha256"]:
        raise ConditionalChoiceArtifactV2Error("raw score hash changed")
    source_bundle = read_extraction_bundle(source_bundle_path)
    if source_bundle["verification"]["archive_sha256"] != descriptor["source_archive_sha256"]:
        raise ConditionalChoiceArtifactV2Error("source archive hash changed")
    if len(source_bundle["sources"]) != 1:
        raise ConditionalChoiceArtifactV2Error("exactly one source manifest is required")
    source = source_bundle["sources"][0]
    if source["source_manifest_sha256"] != descriptor["source_manifest_sha256"]:
        raise ConditionalChoiceArtifactV2Error("source manifest hash changed")
    catalog = load_probe_catalog(catalog_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    raw_rows = _load_jsonl(raw_scores_path)
    return source_bundle, source, catalog, result, raw_rows


def _validate_result_and_rows(
    *, descriptor: Mapping[str, Any], catalog: Mapping[str, Any], result: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any], str]], dict[str, int]]:
    result_payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    if (
        result.get("format") != descriptor["result_format"]
        or result.get("campaign") != descriptor["campaign"]
        or result.get("verdict") not in descriptor["accepted_result_verdicts"]
        or result.get("catalog_sha256") != descriptor["catalog_sha256"]
        or result.get("source_manifest_sha256") != descriptor["source_manifest_sha256"]
        or result.get("evidence_sha256") != descriptor["result_evidence_sha256"]
        or _evidence_sha(result_payload) != result.get("evidence_sha256")
    ):
        raise ConditionalChoiceArtifactV2Error("source result identity or self-hash changed")
    raw_artifact = result.get("artifacts", {}).get("choice_scores", {})
    if raw_artifact.get("sha256") != descriptor["raw_scores_sha256"]:
        raise ConditionalChoiceArtifactV2Error("source result is not bound to raw scores")
    probes = list(catalog.get("probes", []))
    if len(probes) != descriptor["source_rows"] or len(raw_rows) != descriptor["source_rows"]:
        raise ConditionalChoiceArtifactV2Error("catalog/raw cardinality changed")
    by_id = {str(row["probe_id"]): row for row in probes}
    if len(by_id) != len(probes):
        raise ConditionalChoiceArtifactV2Error("catalog probe IDs are duplicated")
    seen: set[str] = set()
    passing = []
    family_passing: Counter[str] = Counter()
    ties = 0
    for index, observation in enumerate(raw_rows):
        probe_id = str(observation.get("probe_id", ""))
        probe = by_id.get(probe_id)
        if probe is None or probe_id in seen:
            raise ConditionalChoiceArtifactV2Error("raw probe is absent or duplicated")
        seen.add(probe_id)
        prompt = str(probe["prompt"])
        candidates = probe.get("conditional_candidates")
        observed_candidates = observation.get("candidate_codes")
        scores = observation.get("candidate_scores")
        expected = probe.get("evaluator", {}).get("value")
        if (
            observation.get("catalog_index") != index
            or observation.get("prompt_sha256") != _text_sha(prompt)
            or probe.get("split") != "search"
            or probe.get("destination_scope") != "english_core"
            or probe.get("capability") != "domain_independent_reasoning"
            or probe.get("domain") != "domain_independent"
            or probe.get("knowledge_class") != "english_linguistic_form"
            or probe.get("content_basis") != "abstract_or_nonce_content"
            or probe.get("domain_labels") != []
            or probe.get("domain_claims") != []
            or probe.get("output_introduces_unsupplied_facts") is not False
            or not isinstance(candidates, list)
            or candidates != observed_candidates
            or len(candidates) != 3
            or len(set(candidates)) != 3
            or expected not in candidates
            or not isinstance(scores, list)
            or len(scores) != 3
        ):
            raise ConditionalChoiceArtifactV2Error("raw row changed its catalog contract")
        means = []
        for score in scores:
            if not isinstance(score, Mapping):
                raise ConditionalChoiceArtifactV2Error("candidate score is malformed")
            mean = float(score.get("mean_log_probability", float("nan")))
            count = score.get("token_count")
            if not math.isfinite(mean) or isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ConditionalChoiceArtifactV2Error("candidate score is invalid")
            means.append(mean)
        selected_index = max(range(3), key=means.__getitem__)
        ordered = sorted(means, reverse=True)
        selected = candidates[selected_index]
        margin = ordered[0] - ordered[1]
        tie = ordered[0] == ordered[1]
        passed = selected == expected
        if (
            observation.get("selected_index") != selected_index
            or observation.get("selected_output") != selected
            or observation.get("selected_output_sha256") != _text_sha(selected)
            or observation.get("expected_output_sha256") != _text_sha(str(expected))
            or observation.get("tie") is not tie
            or observation.get("passed") is not passed
            or not math.isclose(float(observation.get("top_margin_mean_log_probability", float("nan"))), margin, rel_tol=0.0, abs_tol=1e-12)
            or margin <= 0.0
        ):
            raise ConditionalChoiceArtifactV2Error("raw argmax evidence is stale")
        ties += int(tie)
        if passed:
            family = str(observation.get("premise_family"))
            family_passing[family] += 1
            passing.append((probe, observation, _canonical_sha(observation)))
    if seen != set(by_id) or ties:
        raise ConditionalChoiceArtifactV2Error("raw matrix is incomplete or tied")
    recomputed = {
        "rows": len(raw_rows),
        "passing": len(passing),
        "ties": ties,
        "family_passing": dict(sorted(family_passing.items())),
    }
    metrics = result.get("metrics", {})
    if any(metrics.get(key) != value for key, value in recomputed.items()):
        raise ConditionalChoiceArtifactV2Error("source aggregates do not recompute")
    if len(passing) < descriptor["minimum_packaged_rows"]:
        raise ConditionalChoiceArtifactV2Error("too few independently correct selections")
    if set(family_passing) != {str(index) for index in range(7)} or min(family_passing.values()) < descriptor["minimum_packaged_rows_per_family"]:
        raise ConditionalChoiceArtifactV2Error("selective artifact family floor failed")
    if len(passing) != len(raw_rows) and descriptor["allow_selective_packaging"] is not True:
        raise ConditionalChoiceArtifactV2Error("selective packaging was not authorized")
    return passing, dict(sorted(family_passing.items()))


def build_artifact(
    *,
    descriptor_path: Path,
    source_bundle_path: Path,
    catalog_path: Path,
    result_path: Path,
    raw_scores_path: Path,
    domain_ontology_path: Path,
    output_path: Path,
    token_counter: Callable[[str], int] | None = None,
) -> dict[str, Any]:
    receipt_path = output_path.with_name(output_path.name + ".receipt.json")
    if output_path.exists() or receipt_path.exists():
        raise ConditionalChoiceArtifactV2Error(f"immutable artifact exists: {output_path}")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    validate_descriptor(descriptor)
    source_bundle, source, catalog, result, raw_rows = _validate_dependencies(
        descriptor=descriptor,
        source_bundle_path=source_bundle_path,
        catalog_path=catalog_path,
        result_path=result_path,
        raw_scores_path=raw_scores_path,
    )
    passing, family_passing = _validate_result_and_rows(
        descriptor=descriptor, catalog=catalog, result=result, raw_rows=raw_rows
    )
    ontology = json.loads(domain_ontology_path.read_text(encoding="utf-8"))
    validate_domain_ontology(ontology)
    if token_counter is None:
        token_counter = _load_source_token_counter(
            model_id=source["model_id"], revision=source["revision"], trust_remote_code=bool(source["trust_remote_code"])
        )
    records, probe_results = [], []
    for probe, observation, observation_hash in passing:
        output = str(observation["selected_output"])
        record = build_segregated_extraction_record(
            destination_scope="english_core",
            capability="domain_independent_reasoning",
            domain="domain_independent",
            provenance=f"conditional-choice-v2:{result['evidence_sha256']}:{observation_hash}",
            split="search",
            source_model=source["model_id"],
            source_model_revision=source["revision"],
            prompt=str(probe["prompt"]),
            output=output,
            teacher_tokens=int(token_counter(output)),
            teacher_token_counter=COUNTER,
            knowledge_class=str(probe["knowledge_class"]),
            content_basis=str(probe["content_basis"]),
            domain_labels=[],
            domain_claims=[],
            label_method=str(probe["label_method"]),
            label_evidence_sha256=str(probe["label_evidence_sha256"]),
            output_introduces_unsupplied_facts=False,
        )
        margin = float(observation["top_margin_mean_log_probability"])
        evaluator = {
            "kind": "conditional_source_preference",
            "prompt_contract_sha256": _text_sha(str(probe["prompt"])),
            "source_evidence_sha256": result["evidence_sha256"],
            "source_observation_sha256": observation_hash,
            "source_manifest_sha256": source["source_manifest_sha256"],
            "candidate_set_sha256": _canonical_sha(observation["candidate_codes"]),
            "selected_output_sha256": record["output_sha256"],
            "top_margin_mean_log_probability": margin,
            "teacher_generated_output": False,
        }
        probe_result = build_probe_result(
            record=record,
            source_manifest_sha256=source["source_manifest_sha256"],
            probe_id=str(probe["probe_id"]),
            evaluator=evaluator,
            passed=True,
            score=1.0 / (1.0 + math.exp(-margin)),
            seed=int(probe["seed"]),
        )
        records.append(record)
        probe_results.append(probe_result)
    inventory = build_capability_inventory(
        source_manifest=source,
        records=records,
        probe_results=probe_results,
        minimum_distinct_probes=descriptor["minimum_packaged_rows"],
        minimum_pass_rate=1.0,
        minimum_wilson_lower_bound=0.99,
        qualification_splits=("search",),
    )
    if inventory["available_entry_count"] != 1:
        raise ConditionalChoiceArtifactV2Error("selective reasoning inventory did not qualify")
    selection = build_user_selection_plan(
        [inventory],
        include_english_core=True,
        english_capabilities=("domain_independent_reasoning",),
        domains=(),
        source_policy="best_evidence",
        allow_unverified_development_selection=True,
    )
    total_tokens = sum(int(row["teacher_tokens"]) for row in records)
    budgets = build_nested_teacher_budgets(
        records,
        requested_teacher_token_budgets=sorted({max(1, total_tokens // 4), max(1, total_tokens // 2), total_tokens}),
        split="search",
        ordering_seed=f"{catalog['catalog_id']}:conditional-choice-v2",
    )
    unique_outputs = {row["output_sha256"]: (row["output_utf8_bytes"], row["teacher_tokens"]) for row in records}
    accounting = result["accounting"]
    ledger = {
        "schema_version": "abi-source-extraction-ledger/1",
        "status": "SELECTIVE_CONDITIONAL_WEIGHT_TRAINING_MATERIAL_NOT_LAYERCAKE_CERTIFIED",
        "raw_source_prompt_count": len(catalog["probes"]),
        "raw_source_prompt_bytes": sum(len(str(row["prompt"]).encode("utf-8")) for row in catalog["probes"]),
        "unique_prompt_utf8_bytes": sum(len(str(row["prompt"]).encode("utf-8")) for row in catalog["probes"]),
        "teacher_generated_output_bytes": 0,
        "duplicate_adjusted_teacher_output_bytes": 0,
        "conditional_selected_output_bytes": sum(int(row["output_utf8_bytes"]) for row in records),
        "duplicate_adjusted_conditional_selected_output_bytes": sum(int(value[0]) for value in unique_outputs.values()),
        "teacher_tokens": total_tokens,
        "duplicate_adjusted_teacher_tokens": sum(int(value[1]) for value in unique_outputs.values()),
        "teacher_token_counter": COUNTER,
        "logits_stored_count": int(accounting["candidate_scores_stored"]),
        "logits_stored_bytes": int(accounting["candidate_score_storage_bytes_float64_equivalent"]),
        "stored_values_are_scalar_candidate_scores_not_full_logits": True,
        "full_vocabulary_logit_vectors_stored": 0,
        "ephemeral_full_vocabulary_logits_materialized_during_source_scoring": True,
        "hidden_activations_stored_count": 0,
        "hidden_activations_stored_bytes": 0,
        "frozen_source_parameters_copied": 0,
        "frozen_source_parameter_bytes_copied": 0,
        "source_parameter_count_read": int(result["source_parameters_read"]),
        "source_weight_bytes_read": int(source["weight_bytes"]),
        "final_imported_substrate_parameters": 0,
        "bridge_parameters_trained": 0,
        "one_time_source_extraction_seconds": float(accounting["source_load_seconds"]) + float(accounting["source_inference_seconds"]),
        "source_model_inference_seconds": float(accounting["source_inference_seconds"]),
        "per_host_layercake_certification_seconds": None,
        "final_deployed_footprint_bytes": None,
        "final_cpu_inference_seconds": None,
        "artifact_disk_footprint_bytes": "recorded_in_receipt_sidecar",
        "source_extraction_devices": ["cuda"],
        "external_hardware_used": False,
        "external_hardware_description": str(result["hardware"]),
        "source_manifest_sha256": [source["source_manifest_sha256"]],
        "input_extraction_archives": [{"archive_sha256": source_bundle["verification"]["archive_sha256"], "manifest_sha256": source_bundle["verification"]["manifest_sha256"]}],
        "conditional_choice_qualification": {
            "descriptor_sha256": descriptor["descriptor_sha256"],
            "result_file_sha256": descriptor["result_file_sha256"],
            "raw_scores_sha256": descriptor["raw_scores_sha256"],
            "evidence_sha256": result["evidence_sha256"],
            "method": "mean_conditional_log_probability_argmax_over_preregistered_candidates",
            "rows_scored": len(raw_rows),
            "passing_rows_packaged": len(records),
            "failed_rows_excluded": len(raw_rows) - len(records),
            "selective_packaging": len(records) != len(raw_rows),
            "family_passing": family_passing,
            "candidate_scores_stored": int(accounting["candidate_scores_stored"]),
            "source_candidate_tokens_scored": int(accounting["source_candidate_tokens_scored"]),
        },
        "claim_boundary": (
            "Source-weight-selected finite-choice targets only. Failed source choices are "
            "excluded and counted. This is disclosed development training material, not "
            "teacher generation, general English, host transfer, or ABI moonshot proof."
        ),
    }
    segregation = build_core_domain_segregation_manifest(records, domain_ontology=ontology)
    bundle = build_extraction_bundle(
        output_path,
        source_manifests=[source],
        records=records,
        probe_results=probe_results,
        inventories=[inventory],
        selection=selection,
        budgets=budgets,
        ledger=ledger,
        artifact_role=SEGREGATED_TRAINING_ARTIFACT_ROLE,
        domain_ontology=ontology,
        segregation_manifest=segregation,
    )
    verification = verify_extraction_bundle(output_path)
    receipt = {
        "format": "abi-conditional-choice-source-artifact-receipt/2",
        **bundle,
        "verified": verification["verified"],
        "descriptor_sha256": descriptor["descriptor_sha256"],
        "records": len(records),
        "source_scored_rows": len(raw_rows),
        "source_pass_rate": len(records) / len(raw_rows),
        "excluded_failed_rows": len(raw_rows) - len(records),
        "inventory_sha256": inventory["inventory_sha256"],
        "segregation_sha256": segregation["segregation_sha256"],
        "layercake_invoked": False,
        "layercake_training_authorized": True,
        "abi_transfer_proven": False,
        "full_abi_moonshot": "OPEN",
    }
    receipt["receipt_sha256"] = _canonical_sha(receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--raw-scores", required=True, type=Path)
    parser.add_argument("--domain-ontology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = build_artifact(
        descriptor_path=args.descriptor,
        source_bundle_path=args.source_bundle,
        catalog_path=args.catalog,
        result_path=args.result,
        raw_scores_path=args.raw_scores,
        domain_ontology_path=args.domain_ontology,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
