"""Audit whether cached teacher text is adequate evidence for English transfer.

Domain segregation and archive integrity are necessary but do not establish
that a teacher response is complete, prompt-grounded, or useful supervision.
This audit measures those evidence boundaries without opening final-test data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .layercake_host import _canonical_json_bytes, _sha256_file
from .layercake_host_v3 import load_english_training_rows
from .hf_extraction import prompt_contract_sha256


class TeacherArtifactAdequacyAuditError(RuntimeError):
    """Raised when reference-artifact adequacy cannot be audited exactly."""


GENERIC_EVALUATOR_KINDS = frozenset(
    {"all_of", "nonempty", "maximum_characters", "contains_none"}
)
_CLOSING_CHARACTERS = frozenset("\"'”’)]}")
_TERMINAL_CHARACTERS = frozenset(".!?:;")


def _evaluator_leaf_kinds(evaluator: Mapping[str, Any]) -> set[str]:
    kind = str(evaluator.get("kind", ""))
    kinds = {kind}
    rules = evaluator.get("rules")
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, Mapping):
                kinds.update(_evaluator_leaf_kinds(rule))
    return kinds


def _evaluator_is_content_specific(evaluator: Mapping[str, Any]) -> bool:
    """Return true only when a rule checks more than generic output hygiene."""

    kinds = _evaluator_leaf_kinds(evaluator)
    return bool(kinds - GENERIC_EVALUATOR_KINDS)


def _has_terminal_marker(output: str) -> bool:
    """Conservative surface heuristic; not a semantic completeness claim."""

    value = output.rstrip()
    if not value:
        return False
    if value.endswith("```"):
        return True
    while value and value[-1] in _CLOSING_CHARACTERS:
        value = value[:-1].rstrip()
    return bool(value and value[-1] in _TERMINAL_CHARACTERS)


def _catalog_prompt_contracts(
    catalog_paths: Sequence[Path],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    """Return generation ceilings keyed by exact raw-prompt SHA-256.

    Extraction records retain the model's chat-rendered prompt, while the
    functional evaluator is bound to the raw catalog prompt.  Keying by the
    raw-prompt contract preserves that distinction and prevents a valid chat
    template from becoming a false adequacy failure.
    """

    prompt_caps: dict[str, int] = {}
    evidence = []
    for path in catalog_paths:
        try:
            catalog = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TeacherArtifactAdequacyAuditError(
                f"invalid source catalog: {path}"
            ) from exc
        probes = catalog.get("probes") if isinstance(catalog, dict) else None
        if not isinstance(probes, list) or not probes:
            raise TeacherArtifactAdequacyAuditError(
                f"source catalog has no probes: {path}"
            )
        for probe in probes:
            if not isinstance(probe, Mapping):
                raise TeacherArtifactAdequacyAuditError(
                    f"source catalog probe is invalid: {path}"
                )
            prompt = str(probe.get("prompt", ""))
            ceiling = probe.get("max_new_tokens")
            if not prompt or isinstance(ceiling, bool) or not isinstance(ceiling, int):
                raise TeacherArtifactAdequacyAuditError(
                    f"source catalog prompt contract is incomplete: {path}"
                )
            contract = prompt_contract_sha256(prompt)
            prior = prompt_caps.get(contract)
            if prior is not None and prior != ceiling:
                raise TeacherArtifactAdequacyAuditError(
                    "one source prompt has conflicting generation ceilings"
                )
            prompt_caps[contract] = ceiling
        evidence.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "probes": len(probes),
            }
        )
    return prompt_caps, evidence


def _summarize_records(
    *,
    rows: Sequence[Mapping[str, Any]],
    original_records: Mapping[str, Mapping[str, Any]],
    probe_results: Mapping[str, Mapping[str, Any]],
    prompt_caps: Mapping[str, int],
) -> dict[str, Any]:
    if not rows:
        raise TeacherArtifactAdequacyAuditError("selected budget has no rows")
    response_hashes: Counter[str] = Counter()
    prompt_hashes: Counter[str] = Counter()
    evaluator_signatures: Counter[str] = Counter()
    by_capability: dict[str, Counter[str]] = defaultdict(Counter)
    ceiling_saturated = 0
    ceiling_saturated_without_terminal = 0
    content_specific = 0
    finish_reason_present = 0
    generated_ids_present = 0
    contrastive_rows = 0
    valid_contrastive_rows = 0
    contrastive_evidence_hashes: set[str] = set()
    length_terminated = 0
    prompt_contract_bindings_valid = 0
    content_specific_signatures: Counter[str] = Counter()
    examples = []
    for row in rows:
        record_id = str(row["record_id"])
        original = original_records.get(record_id)
        result = probe_results.get(record_id)
        if original is None or result is None:
            raise TeacherArtifactAdequacyAuditError(
                f"selected row lacks exact source evidence: {record_id}"
            )
        prompt = str(row["prompt"])
        evaluator = result.get("evaluator")
        if not isinstance(evaluator, Mapping):
            raise TeacherArtifactAdequacyAuditError(
                f"selected row lacks an evaluator: {record_id}"
            )
        evaluator_contract = evaluator.get("prompt_contract_sha256")
        rendered_contract = prompt_contract_sha256(prompt)
        if evaluator_contract in prompt_caps:
            ceiling = int(prompt_caps[str(evaluator_contract)])
            bound_prompt_hash = str(evaluator_contract)
            binding_valid = True
        elif prompt in prompt_caps:
            # Backward-compatible support for direct unit inputs and old
            # callers that supplied raw prompts rather than prompt hashes.
            ceiling = int(prompt_caps[prompt])
            bound_prompt_hash = rendered_contract
            binding_valid = evaluator_contract == rendered_contract
        elif rendered_contract in prompt_caps:
            ceiling = int(prompt_caps[rendered_contract])
            bound_prompt_hash = rendered_contract
            binding_valid = evaluator_contract == rendered_contract
        else:
            raise TeacherArtifactAdequacyAuditError(
                f"selected prompt lacks a bound generation ceiling: {record_id}"
            )
        output = str(row["response"])
        capability = str(row["capability"])
        teacher_tokens = int(row["teacher_tokens"])
        if teacher_tokens > ceiling:
            raise TeacherArtifactAdequacyAuditError(
                f"teacher tokens exceed the bound generation ceiling: {record_id}"
            )
        evaluator_signature = hashlib.sha256(
            _canonical_json_bytes(evaluator)
        ).hexdigest()
        evaluator_signatures[evaluator_signature] += 1
        prompt_contract_bindings_valid += int(binding_valid)
        specific = _evaluator_is_content_specific(evaluator)
        if specific:
            content_specific_signatures[evaluator_signature] += 1
        is_contrastive = (
            original.get("teacher_token_counter")
            == "authoritative_source_tokenizer_posthoc_on_contrastive_selection"
        )
        valid_contrastive = (
            is_contrastive
            and evaluator.get("kind")
            == "counterbalanced_source_preference"
            and evaluator.get("teacher_generated_output") is False
            and evaluator.get("selected_output_sha256")
            == hashlib.sha256(output.encode("utf-8")).hexdigest()
            and isinstance(evaluator.get("contrastive_evidence_sha256"), str)
            and len(str(evaluator.get("contrastive_evidence_sha256"))) == 64
            and isinstance(
                evaluator.get("contrastive_observation_sha256"), str
            )
            and len(str(evaluator.get("contrastive_observation_sha256"))) == 64
            and float(evaluator.get("ab_margin", 0.0)) > 0.0
            and float(evaluator.get("ba_margin", 0.0)) > 0.0
        )
        if valid_contrastive:
            contrastive_evidence_hashes.add(
                str(evaluator["contrastive_evidence_sha256"])
            )
        terminal = _has_terminal_marker(output)
        saturated = not is_contrastive and teacher_tokens == ceiling
        has_finish_reason = isinstance(original.get("finish_reason"), str)
        has_generated_ids = isinstance(
            original.get("authoritative_generated_token_ids"), list
        )
        terminated_by_length = original.get("finish_reason") == "length"
        ceiling_saturated += int(saturated)
        ceiling_saturated_without_terminal += int(saturated and not terminal)
        content_specific += int(specific)
        finish_reason_present += int(has_finish_reason)
        generated_ids_present += int(has_generated_ids)
        contrastive_rows += int(is_contrastive)
        valid_contrastive_rows += int(valid_contrastive)
        length_terminated += int(terminated_by_length)
        prompt_hashes[bound_prompt_hash] += 1
        response_hashes[hashlib.sha256(output.encode("utf-8")).hexdigest()] += 1
        summary = by_capability[capability]
        summary["records"] += 1
        summary["teacher_tokens"] += teacher_tokens
        summary["ceiling_saturated"] += int(saturated)
        summary["ceiling_saturated_without_terminal_marker"] += int(
            saturated and not terminal
        )
        summary["content_specific_evaluators"] += int(specific)
        summary["finish_reason_present"] += int(has_finish_reason)
        summary["contrastive_source_selected"] += int(is_contrastive)
        summary["valid_contrastive_source_evidence"] += int(
            valid_contrastive
        )
        summary["length_terminated"] += int(terminated_by_length)
        if saturated and len(examples) < 20:
            examples.append(
                {
                    "record_id": record_id,
                    "capability": capability,
                    "teacher_tokens": teacher_tokens,
                    "generation_ceiling": ceiling,
                    "terminal_marker_present": terminal,
                    "output_sha256": hashlib.sha256(
                        output.encode("utf-8")
                    ).hexdigest(),
                }
            )
    count = len(rows)
    capability_summary = {}
    for capability, values in sorted(by_capability.items()):
        records = values["records"]
        capability_summary[capability] = {
            **dict(values),
            "ceiling_saturation_rate": values["ceiling_saturated"] / records,
            "content_specific_evaluator_rate": (
                values["content_specific_evaluators"] / records
            ),
        }
    return {
        "records": count,
        "unique_prompt_hashes": len(prompt_hashes),
        "duplicate_prompt_instances": count - len(prompt_hashes),
        "unique_response_hashes": len(response_hashes),
        "duplicate_response_instances": count - len(response_hashes),
        "teacher_tokens": sum(int(row["teacher_tokens"]) for row in rows),
        "generation_ceiling_saturated": ceiling_saturated,
        "generation_ceiling_saturation_rate": ceiling_saturated / count,
        "ceiling_saturated_without_terminal_marker": (
            ceiling_saturated_without_terminal
        ),
        "content_specific_evaluators": content_specific,
        "content_specific_evaluator_rate": content_specific / count,
        "generic_hygiene_only_evaluators": count - content_specific,
        "finish_reason_present": finish_reason_present,
        "authoritative_generated_token_ids_present": generated_ids_present,
        "contrastive_source_selected_rows": contrastive_rows,
        "valid_contrastive_source_evidence_rows": valid_contrastive_rows,
        "distinct_contrastive_evidence_sha256": sorted(
            contrastive_evidence_hashes
        ),
        "rows_with_authoritative_generation_or_contrastive_evidence": (
            generated_ids_present + valid_contrastive_rows
        ),
        "length_terminated": length_terminated,
        "prompt_contract_bindings_valid": prompt_contract_bindings_valid,
        "distinct_content_specific_evaluator_signatures": len(
            content_specific_signatures
        ),
        "distinct_evaluator_signatures": len(evaluator_signatures),
        "dominant_evaluator_signature_count": (
            evaluator_signatures.most_common(1)[0][1]
        ),
        "by_capability": capability_summary,
        "disclosed_saturated_examples": examples,
    }


def audit_teacher_artifact_adequacy(
    *,
    bundle_path: str | Path,
    catalog_paths: Sequence[str | Path],
    budget_index: int,
    output_path: str | Path,
    require_semantic_qualification: bool = False,
) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve()
    catalogs = [Path(path).resolve() for path in catalog_paths]
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise TeacherArtifactAdequacyAuditError(
            f"adequacy evidence is immutable: {output_path}"
        )
    rows, budget, bundle = load_english_training_rows(
        bundle_path, budget_index=budget_index
    )
    prompt_caps, catalog_evidence = _catalog_prompt_contracts(catalogs)
    original_records = {
        str(record["record_id"]): record for record in bundle["records"]
    }
    probe_results = {
        str(result["record_id"]): result for result in bundle["probe_results"]
    }
    if len(original_records) != len(bundle["records"]):
        raise TeacherArtifactAdequacyAuditError("duplicate record identifier")
    if len(probe_results) != len(bundle["probe_results"]):
        raise TeacherArtifactAdequacyAuditError("duplicate probe-result identifier")
    summary = _summarize_records(
        rows=rows,
        original_records=original_records,
        probe_results=probe_results,
        prompt_caps=prompt_caps,
    )
    ledger = bundle["ledger"]
    selection = bundle["selection"]
    semantic = ledger.get("semantic_qualification")
    contrastive = ledger.get("contrastive_qualification")
    semantic_runtime = (
        semantic.get("judge_runtime", {})
        if isinstance(semantic, Mapping)
        else {}
    )
    contrastive_hash = (
        contrastive.get("evidence_sha256")
        if isinstance(contrastive, Mapping)
        else None
    )
    contrastive_rows = summary["contrastive_source_selected_rows"]
    generated_rows = summary["records"] - contrastive_rows
    checks = {
        "all_selected_prompts_have_bound_generation_ceiling": True,
        "all_selected_rows_have_generation_finish_or_contrastive_evidence": (
            summary["finish_reason_present"]
            + summary["valid_contrastive_source_evidence_rows"]
            == summary["records"]
        ),
        "all_selected_rows_retain_generated_ids_or_contrastive_evidence": (
            summary["authoritative_generated_token_ids_present"]
            + summary["valid_contrastive_source_evidence_rows"]
            == summary["records"]
        ),
        "all_generated_rows_retain_authoritative_runtime": (
            summary["finish_reason_present"] == generated_rows
            and summary["authoritative_generated_token_ids_present"]
            == generated_rows
        ),
        "all_contrastive_rows_have_valid_source_preference_evidence": (
            summary["valid_contrastive_source_evidence_rows"]
            == contrastive_rows
        ),
        "contrastive_qualification_accounted": (
            contrastive_rows == 0
            or (
                isinstance(contrastive, Mapping)
                and contrastive_hash is not None
                and summary["distinct_contrastive_evidence_sha256"]
                == [contrastive_hash]
                and int(contrastive.get("completion_tokens_scored", 0)) > 0
                and isinstance(contrastive.get("checks"), Mapping)
                and all(bool(value) for value in contrastive["checks"].values())
            )
        ),
        "all_selected_rows_have_content_specific_quality_evaluator": (
            summary["content_specific_evaluators"] == summary["records"]
        ),
        "all_selected_rows_have_prompt_specific_evaluator_contracts": (
            summary["distinct_content_specific_evaluator_signatures"]
            == summary["records"]
        ),
        "all_selected_evaluators_bind_exact_source_prompt": (
            summary["prompt_contract_bindings_valid"] == summary["records"]
        ),
        "no_selected_row_is_length_terminated": (
            summary["length_terminated"] == 0
        ),
        "source_model_inference_time_recorded": (
            float(ledger.get("source_model_inference_seconds", 0.0)) > 0.0
        ),
        "one_time_source_extraction_time_recorded": (
            float(ledger.get("one_time_source_extraction_seconds", 0.0)) > 0.0
        ),
        "source_extraction_device_recorded": bool(
            ledger.get("source_extraction_devices")
        ),
        "selection_is_promotion_eligible": (
            selection.get("promotion_eligible") is True
            and selection.get("allow_unverified_development_selection") is False
        ),
        "at_least_100_records_per_capability": all(
            values["records"] >= 100
            for values in summary["by_capability"].values()
        ),
        "required_semantic_qualification_accounted": (
            not require_semantic_qualification
            or (
                generated_rows == 0
                or (
                isinstance(semantic, Mapping)
                and int(semantic.get("judge_generated_tokens", 0)) > 0
                and float(semantic.get("judge_load_seconds", 0.0)) > 0.0
                and float(semantic.get("judge_inference_seconds", 0.0)) > 0.0
                and semantic_runtime.get("device") == "cuda"
                and semantic_runtime.get("weight_execution_precision")
                == "bitsandbytes_int8"
                and semantic_runtime.get("cpu_offload_enabled") is False
                and isinstance(semantic.get("judge_durable_journal"), Mapping)
                and bool(
                    semantic.get("judge_durable_journal", {}).get("sha256")
                )
                )
            )
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    evidence: dict[str, Any] = {
        "format": "abi-teacher-artifact-adequacy-audit/2",
        "status": (
            "PASS_REFERENCE_ARTIFACT_ADEQUACY"
            if not failures
            else "FAIL_INCOMPLETE_REFERENCE_ARTIFACT_ADEQUACY_EVIDENCE"
        ),
        "bundle": {
            "path": str(bundle_path),
            "sha256": _sha256_file(bundle_path),
            "manifest_sha256": bundle["verification"]["manifest_sha256"],
            "budget_index": budget_index,
            "budget_id": budget["budget_id"],
            "split": budget["split"],
        },
        "catalogs": catalog_evidence,
        "summary": summary,
        "source_accounting": {
            key: ledger.get(key)
            for key in (
                "teacher_tokens",
                "teacher_generated_output_bytes",
                "raw_source_prompt_count",
                "unique_prompt_utf8_bytes",
                "source_model_inference_seconds",
                "one_time_source_extraction_seconds",
                "source_extraction_devices",
                "external_hardware_used",
                "external_hardware_description",
                "source_parameter_count_read",
                "source_weight_bytes_read",
                "semantic_qualification",
                "contrastive_qualification",
                "teacher_token_counters",
                "logits_stored_count",
                "logits_stored_bytes",
                "ephemeral_full_logit_elements_materialized",
            )
        },
        "selection_boundary": {
            "promotion_eligible": selection.get("promotion_eligible"),
            "allow_unverified_development_selection": selection.get(
                "allow_unverified_development_selection"
            ),
        },
        "checks": checks,
        "failures": failures,
        "interpretation": (
            "A source response passing generic output-hygiene rules is not "
            "evidence that it completed the request or preserved supplied "
            "details. Ceiling saturation is reported as a measured risk, not "
            "as proof of truncation; absent finish reasons make that risk "
            "unresolvable from this artifact."
        ),
        "successor_requirements": [
            "retain authoritative generated token IDs and finish reasons for generated rows, or exact counterbalanced source-preference evidence for contrastive rows",
            "reject ceiling-terminated responses unless independently completed and revalidated",
            "use prompt-specific functional evaluators for every selected response",
            "record source inference time and exact extraction hardware",
            "provide at least 100 distinct passing search records per capability",
            "remain domain-segregated and exclude final-test data",
        ],
        "training_authorized": not failures,
        "semantic_qualification_required": require_semantic_qualification,
        "final_test_accessed": False,
        "promotion_eligible": False,
        "abi_transfer_proven": False,
        "moonshot_complete": False,
    }
    evidence["evidence_sha256"] = hashlib.sha256(
        _canonical_json_bytes(evidence)
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--catalog", action="append", required=True)
    parser.add_argument("--budget-index", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-semantic-qualification", action="store_true")
    args = parser.parse_args(argv)
    evidence = audit_teacher_artifact_adequacy(
        bundle_path=args.bundle,
        catalog_paths=args.catalog,
        budget_index=args.budget_index,
        output_path=args.output,
        require_semantic_qualification=args.require_semantic_qualification,
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "summary": evidence["summary"],
                "failures": evidence["failures"],
                "evidence_sha256": evidence["evidence_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not evidence["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
