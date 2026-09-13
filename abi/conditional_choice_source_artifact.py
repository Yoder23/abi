"""Package source-weight-selected outputs as segregated ABI material.

This path is intentionally distinct from sequence distillation.  The source
does not generate a response.  Instead, a preregistered finite candidate set
is scored under frozen source weights and the independently evaluated argmax
is retained with its complete scalar-score evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from .capability_pipeline import (
    SEGREGATED_TRAINING_ARTIFACT_ROLE,
    build_capability_inventory,
    build_extraction_bundle,
    build_inventory_survey_plan,
    build_nested_teacher_budgets,
    build_probe_result,
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


COUNTER = "authoritative_source_tokenizer_posthoc_on_conditional_choice_selection"
RESULT_FORMAT = "abi-r72-robust-conditional-choice-weight-interrogation/1"
EXPECTED_VERDICT = "PASS_R72_AUTHORIZE_SEGREGATED_PACKAGING"
EXPECTED_SOURCE_ARCHIVE_SHA256 = (
    "82d1ab8a3ee7b4aa351b5c74b4a229d75e845313047065780227e8e403363150"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "3bac528a1825e77dcb35963f5c78946fb14400e1cd832e0db40c2d964360c310"
)
EXPECTED_CATALOG_SHA256 = (
    "d179445c92a649f5ab6587c1aa71e00e7c622c326f8ea84438497542c00c2b47"
)
EXPECTED_RESULT_FILE_SHA256 = (
    "fcbdf70621bb5c5335945c6c16e7f46bc5138c077626c2734f8e1eb0925dd38c"
)
EXPECTED_RAW_SHA256 = (
    "5c8f8a655800612f495b20b63d3bf3dcf82dbef49c5a6eb0f22659a3f47d4e4b"
)
EXPECTED_EVIDENCE_SHA256 = (
    "81f25c0725a73f719b4e242f29ec93fcd49bdc8fc212e6f6bc78e4bc64c4005f"
)
EXPECTED_ROWS = 2_100
EXPECTED_PASSING = 2_098


class ConditionalChoiceArtifactError(RuntimeError):
    """Raised when conditional source evidence cannot be packaged exactly."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _r72_evidence_sha(value: Mapping[str, Any]) -> str:
    """Reproduce the newline-terminated canonical hash frozen by R72."""

    raw = (json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConditionalChoiceArtifactError(
                    f"invalid score JSONL at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ConditionalChoiceArtifactError("score row is not an object")
            rows.append(value)
    return rows


def _validate_result(result: Mapping[str, Any], result_path: Path, raw_path: Path) -> None:
    if _file_sha(result_path) != EXPECTED_RESULT_FILE_SHA256:
        raise ConditionalChoiceArtifactError("R72 result file identity changed")
    if _file_sha(raw_path) != EXPECTED_RAW_SHA256:
        raise ConditionalChoiceArtifactError("R72 raw-score identity changed")
    if (
        result.get("format") != RESULT_FORMAT
        or result.get("verdict") != EXPECTED_VERDICT
        or result.get("evidence_sha256") != EXPECTED_EVIDENCE_SHA256
    ):
        raise ConditionalChoiceArtifactError("R72 result is not the frozen pass")
    payload = {key: value for key, value in result.items() if key != "evidence_sha256"}
    if _r72_evidence_sha(payload) != result["evidence_sha256"]:
        raise ConditionalChoiceArtifactError("R72 result self-hash is stale")
    artifact = result.get("artifacts", {}).get("choice_scores", {})
    if (
        artifact.get("sha256") != EXPECTED_RAW_SHA256
        or int(artifact.get("bytes", -1)) != raw_path.stat().st_size
        or result.get("catalog_sha256") != EXPECTED_CATALOG_SHA256
        or result.get("source_manifest_sha256") != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise ConditionalChoiceArtifactError("R72 result dependency binding changed")
    metrics = result.get("metrics", {})
    if (
        int(metrics.get("rows", -1)) != EXPECTED_ROWS
        or int(metrics.get("passing", -1)) != EXPECTED_PASSING
        or int(metrics.get("ties", -1)) != 0
        or not all(result.get("gates", {}).values())
    ):
        raise ConditionalChoiceArtifactError("R72 aggregate gates no longer recompute")


def _validated_rows(
    *, catalog: Mapping[str, Any], raw_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any], str]], dict[str, int]]:
    probes = list(catalog.get("probes", []))
    probes_by_id = {str(probe.get("probe_id")): probe for probe in probes}
    if (
        len(probes) != EXPECTED_ROWS
        or len(probes_by_id) != EXPECTED_ROWS
        or len(raw_rows) != EXPECTED_ROWS
    ):
        raise ConditionalChoiceArtifactError("R72 row cardinality changed")
    seen: set[str] = set()
    passing: list[tuple[Mapping[str, Any], Mapping[str, Any], str]] = []
    ties = 0
    per_family: dict[str, int] = {}
    for catalog_index, observation in enumerate(raw_rows):
        probe_id = str(observation.get("probe_id", ""))
        probe = probes_by_id.get(probe_id)
        if probe is None or probe_id in seen:
            raise ConditionalChoiceArtifactError("score probe is missing or duplicated")
        seen.add(probe_id)
        prompt = str(probe["prompt"])
        expected_values = probe.get("evaluator", {}).get("values", [])
        candidates = observation.get("candidate_codes")
        scores = observation.get("candidate_scores")
        if (
            observation.get("catalog_index") != catalog_index
            or observation.get("prompt_sha256") != _text_sha(prompt)
            or probe.get("split") != "search"
            or probe.get("destination_scope") != "english_core"
            or probe.get("capability") != "domain_independent_reasoning"
            or probe.get("knowledge_class") != "english_linguistic_form"
            or probe.get("content_basis") != "abstract_or_nonce_content"
            or probe.get("domain_labels") != []
            or probe.get("domain_claims") != []
            or probe.get("output_introduces_unsupplied_facts") is not False
            or not isinstance(expected_values, list)
            or len(expected_values) != 1
            or not isinstance(candidates, list)
            or len(candidates) != 3
            or len(set(candidates)) != 3
            or not isinstance(scores, list)
            or len(scores) != 3
        ):
            raise ConditionalChoiceArtifactError("score row changed its catalog contract")
        means: list[float] = []
        for score in scores:
            if not isinstance(score, Mapping):
                raise ConditionalChoiceArtifactError("candidate score is malformed")
            mean = float(score.get("mean_log_probability", float("nan")))
            token_count = score.get("token_count")
            if (
                not math.isfinite(mean)
                or isinstance(token_count, bool)
                or not isinstance(token_count, int)
                or token_count <= 0
            ):
                raise ConditionalChoiceArtifactError("candidate score is non-finite")
            means.append(mean)
        selected_index = max(range(3), key=means.__getitem__)
        ordered = sorted(means, reverse=True)
        selected = str(candidates[selected_index])
        expected = str(expected_values[0])
        tie = ordered[0] == ordered[1]
        passed = selected == expected
        margin = ordered[0] - ordered[1]
        if (
            observation.get("selected_index") != selected_index
            or observation.get("selected_output") != selected
            or observation.get("selected_output_sha256") != _text_sha(selected)
            or observation.get("expected_output_sha256") != _text_sha(expected)
            or observation.get("tie") is not tie
            or observation.get("passed") is not passed
            or not math.isclose(
                float(observation.get("top_margin_mean_log_probability", float("nan"))),
                margin,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or margin <= 0.0
        ):
            raise ConditionalChoiceArtifactError("score argmax or margin is stale")
        ties += int(tie)
        family = str(observation.get("premise_family"))
        if passed:
            observation_hash = _canonical_sha(observation)
            passing.append((probe, observation, observation_hash))
            per_family[family] = per_family.get(family, 0) + 1
    if seen != set(probes_by_id) or len(passing) != EXPECTED_PASSING or ties:
        raise ConditionalChoiceArtifactError("R72 exact pass set changed")
    if set(per_family) != {str(index) for index in range(7)} or min(per_family.values()) < 298:
        raise ConditionalChoiceArtifactError("R72 family floor changed")
    return passing, per_family


def _budgets(records: Sequence[Mapping[str, Any]], ordering_seed: str) -> list[dict[str, Any]]:
    total = sum(int(row["teacher_tokens"]) for row in records)
    requested = sorted({max(1, total // 4), max(1, total // 2), total})
    return build_nested_teacher_budgets(
        records,
        requested_teacher_token_budgets=requested,
        split="search",
        ordering_seed=ordering_seed,
    )


def build_conditional_choice_artifact(
    *,
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
        raise ConditionalChoiceArtifactError(f"immutable R73 output exists: {output_path}")
    source_bundle = read_extraction_bundle(source_bundle_path)
    if source_bundle["verification"]["archive_sha256"] != EXPECTED_SOURCE_ARCHIVE_SHA256:
        raise ConditionalChoiceArtifactError("source bundle identity changed")
    if len(source_bundle["sources"]) != 1:
        raise ConditionalChoiceArtifactError("R73 requires exactly one source")
    source = source_bundle["sources"][0]
    if source["source_manifest_sha256"] != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ConditionalChoiceArtifactError("source manifest identity changed")
    if _file_sha(catalog_path) != EXPECTED_CATALOG_SHA256:
        raise ConditionalChoiceArtifactError("R72 catalog identity changed")
    catalog = load_probe_catalog(catalog_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate_result(result, result_path, raw_scores_path)
    raw_rows = _load_jsonl(raw_scores_path)
    passing, per_family = _validated_rows(catalog=catalog, raw_rows=raw_rows)
    ontology = json.loads(domain_ontology_path.read_text(encoding="utf-8"))
    validate_domain_ontology(ontology)
    if token_counter is None:
        token_counter = _load_source_token_counter(
            model_id=source["model_id"],
            revision=source["revision"],
            trust_remote_code=bool(source["trust_remote_code"]),
        )

    records: list[dict[str, Any]] = []
    probe_results: list[dict[str, Any]] = []
    for probe, observation, observation_hash in passing:
        output = str(observation["selected_output"])
        record = build_segregated_extraction_record(
            destination_scope="english_core",
            capability="domain_independent_reasoning",
            domain="domain_independent",
            provenance=f"conditional-choice:{EXPECTED_EVIDENCE_SHA256}:{observation_hash}",
            split="search",
            source_model=source["model_id"],
            source_model_revision=source["revision"],
            prompt=str(probe["prompt"]),
            output=output,
            teacher_tokens=int(token_counter(output)),
            teacher_token_counter=COUNTER,
            knowledge_class=str(probe["knowledge_class"]),
            content_basis=str(probe["content_basis"]),
            domain_labels=list(probe["domain_labels"]),
            domain_claims=list(probe["domain_claims"]),
            label_method=str(probe["label_method"]),
            label_evidence_sha256=str(probe["label_evidence_sha256"]),
            output_introduces_unsupplied_facts=False,
        )
        margin = float(observation["top_margin_mean_log_probability"])
        evaluator = {
            "kind": "conditional_source_preference",
            "prompt_contract_sha256": _text_sha(str(probe["prompt"])),
            "conditional_choice_evidence_sha256": EXPECTED_EVIDENCE_SHA256,
            "conditional_choice_observation_sha256": observation_hash,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "candidate_set_sha256": _canonical_sha(observation["candidate_codes"]),
            "selected_output_sha256": record["output_sha256"],
            "top_margin_mean_log_probability": margin,
            "teacher_generated_output": False,
        }
        probe_result = build_probe_result(
            record=record,
            source_manifest_sha256=EXPECTED_SOURCE_MANIFEST_SHA256,
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
        minimum_distinct_probes=1_995,
        minimum_pass_rate=0.95,
        minimum_wilson_lower_bound=0.95,
        qualification_splits=("search",),
    )
    if inventory["available_entry_count"] != 1:
        raise ConditionalChoiceArtifactError("reasoning inventory did not qualify")
    selection = build_inventory_survey_plan(inventory)
    unique_outputs = {
        str(row["output_sha256"]): (int(row["output_utf8_bytes"]), int(row["teacher_tokens"]))
        for row in records
    }
    accounting = result["accounting"]
    raw_prompt_bytes = sum(len(str(probe["prompt"]).encode("utf-8")) for probe in catalog["probes"])
    ledger = {
        "schema_version": "abi-source-extraction-ledger/1",
        "status": "CONDITIONAL_WEIGHT_SELECTED_TRAINING_MATERIAL_NOT_LAYERCAKE_CERTIFIED",
        "raw_source_prompt_count": EXPECTED_ROWS,
        "raw_source_prompt_bytes": raw_prompt_bytes,
        "unique_prompt_utf8_bytes": raw_prompt_bytes,
        "teacher_generated_output_bytes": 0,
        "duplicate_adjusted_teacher_output_bytes": 0,
        "conditional_selected_output_bytes": sum(int(row["output_utf8_bytes"]) for row in records),
        "duplicate_adjusted_conditional_selected_output_bytes": sum(value[0] for value in unique_outputs.values()),
        "teacher_tokens": sum(int(row["teacher_tokens"]) for row in records),
        "duplicate_adjusted_teacher_tokens": sum(value[1] for value in unique_outputs.values()),
        "teacher_token_counter": COUNTER,
        "logits_stored_count": int(accounting["candidate_scores_stored"]),
        "logits_stored_bytes": int(accounting["candidate_score_storage_bytes_float64_equivalent"]),
        "ephemeral_full_logit_elements_materialized": 0,
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
        "source_inference_runtimes": [{"framework": "transformers", "dtype": "bfloat16", "attention": "eager"}],
        "external_hardware_used": False,
        "external_hardware_description": str(result["hardware"]),
        "source_manifest_sha256": [EXPECTED_SOURCE_MANIFEST_SHA256],
        "input_extraction_archives": [{"archive_sha256": EXPECTED_SOURCE_ARCHIVE_SHA256, "manifest_sha256": source_bundle["verification"]["manifest_sha256"]}],
        "conditional_choice_qualification": {
            "evidence_file_sha256": EXPECTED_RESULT_FILE_SHA256,
            "raw_scores_sha256": EXPECTED_RAW_SHA256,
            "evidence_sha256": EXPECTED_EVIDENCE_SHA256,
            "method": "mean_conditional_log_probability_argmax_over_preregistered_candidates",
            "rows_scored": EXPECTED_ROWS,
            "passing_rows_packaged": EXPECTED_PASSING,
            "family_passing": per_family,
            "candidate_scores_stored": int(accounting["candidate_scores_stored"]),
            "source_candidate_tokens_scored": int(accounting["source_candidate_tokens_scored"]),
        },
        "claim_boundary": (
            "This artifact contains source-weight-selected conditional-choice targets, "
            "not teacher-generated text. It is bounded development training material "
            "and does not certify LayerCake transfer, general English, or the ABI moonshot."
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
        budgets=_budgets(records, f"{catalog['catalog_id']}:conditional-choice-r73"),
        ledger=ledger,
        artifact_role=SEGREGATED_TRAINING_ARTIFACT_ROLE,
        domain_ontology=ontology,
        segregation_manifest=segregation,
    )
    verification = verify_extraction_bundle(output_path)
    receipt: dict[str, Any] = {
        "format": "abi-conditional-choice-source-artifact-receipt/1",
        **bundle,
        "verified": verification["verified"],
        "records": len(records),
        "source_scored_rows": EXPECTED_ROWS,
        "source_pass_rate": EXPECTED_PASSING / EXPECTED_ROWS,
        "conditional_choice_evidence_sha256": EXPECTED_EVIDENCE_SHA256,
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
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--raw-scores", required=True, type=Path)
    parser.add_argument("--domain-ontology", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    receipt = build_conditional_choice_artifact(
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
