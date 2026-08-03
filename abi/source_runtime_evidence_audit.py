"""Audit source-survey runtime evidence before an acquisition composition."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .capability_pipeline import read_extraction_bundle
from .layercake_host import _canonical_json_bytes, _sha256_file


class SourceRuntimeEvidenceAuditError(RuntimeError):
    """Raised when source runtime evidence cannot be audited exactly."""


def _summarize_runtime_records(
    records: Sequence[Mapping[str, Any]],
    probe_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not records:
        raise SourceRuntimeEvidenceAuditError("source survey has no records")
    result_by_record = {
        str(result["record_id"]): result for result in probe_results
    }
    if len(result_by_record) != len(probe_results):
        raise SourceRuntimeEvidenceAuditError("duplicate probe-result record ID")
    finish_reasons: Counter[str] = Counter()
    by_capability: dict[str, Counter[str]] = {}
    runtime_complete = 0
    id_count_matches = 0
    positive_input_counts = 0
    passing = 0
    for record in records:
        record_id = str(record["record_id"])
        result = result_by_record.get(record_id)
        if result is None:
            raise SourceRuntimeEvidenceAuditError(
                f"record lacks a bound probe result: {record_id}"
            )
        ids = record.get("authoritative_generated_token_ids")
        finish_reason = record.get("finish_reason")
        maximum = record.get("generation_max_new_tokens")
        input_tokens = record.get("teacher_input_tokens")
        complete = (
            isinstance(ids, list)
            and isinstance(finish_reason, str)
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
        )
        runtime_complete += int(complete)
        id_count_matches += int(
            isinstance(ids, list) and len(ids) == int(record["teacher_tokens"])
        )
        positive_input_counts += int(
            isinstance(input_tokens, int)
            and not isinstance(input_tokens, bool)
            and input_tokens > 0
        )
        finish_reasons[str(finish_reason)] += 1
        passed = result.get("passed") is True
        passing += int(passed)
        capability = str(record["capability"])
        row = by_capability.setdefault(capability, Counter())
        row["records"] += 1
        row["passing"] += int(passed)
        row["length_terminated"] += int(finish_reason == "length")
    return {
        "records": len(records),
        "unique_record_ids": len({str(record["record_id"]) for record in records}),
        "runtime_evidence_complete": runtime_complete,
        "authoritative_id_count_matches": id_count_matches,
        "positive_input_token_counts": positive_input_counts,
        "finish_reasons": dict(sorted(finish_reasons.items())),
        "length_terminated": finish_reasons.get("length", 0),
        "eos_terminated": finish_reasons.get("eos_token", 0),
        "passing_probe_results": passing,
        "failing_probe_results": len(records) - passing,
        "by_capability": {
            capability: dict(values)
            for capability, values in sorted(by_capability.items())
        },
    }


def audit_source_runtime_evidence(
    *,
    bundle_path: str | Path,
    output_path: str | Path,
    expected_records: int,
    require_cuda: bool = True,
    minimum_functional_passes: int | None = None,
    required_inference_precision: str | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise SourceRuntimeEvidenceAuditError(
            f"runtime evidence is immutable: {output_path}"
        )
    bundle = read_extraction_bundle(bundle_path)
    summary = _summarize_runtime_records(
        bundle["records"], bundle["probe_results"]
    )
    if minimum_functional_passes is not None and (
        isinstance(minimum_functional_passes, bool)
        or not isinstance(minimum_functional_passes, int)
        or not 0 <= minimum_functional_passes <= expected_records
    ):
        raise SourceRuntimeEvidenceAuditError(
            "minimum functional passes must be within the expected record count"
        )
    ledger = bundle["ledger"]
    devices = [str(value) for value in ledger.get("source_extraction_devices", [])]
    runtimes = [
        dict(value) for value in ledger.get("source_inference_runtimes", [])
    ]
    checks = {
        "verified_source_survey_vault": (
            bundle["verification"].get("verified") is True
            and bundle["verification"].get("artifact_role")
            == "source_capability_survey_vault"
        ),
        "exact_expected_record_count": summary["records"] == expected_records,
        "unique_record_ids": summary["unique_record_ids"] == summary["records"],
        "all_runtime_evidence_complete": (
            summary["runtime_evidence_complete"] == summary["records"]
        ),
        "all_authoritative_id_counts_match": (
            summary["authoritative_id_count_matches"] == summary["records"]
        ),
        "all_input_token_counts_positive": (
            summary["positive_input_token_counts"] == summary["records"]
        ),
        "all_rows_eos_terminated": summary["eos_terminated"] == summary["records"],
        "no_length_terminated_rows": summary["length_terminated"] == 0,
        "source_inference_time_recorded": (
            float(ledger.get("source_model_inference_seconds", 0.0)) > 0.0
        ),
        "total_extraction_time_recorded": (
            float(ledger.get("one_time_source_extraction_seconds", 0.0)) > 0.0
        ),
        "source_device_recorded": bool(devices),
        "required_cuda_recorded": (not require_cuda or devices == ["cuda"]),
        "required_inference_precision_recorded": (
            required_inference_precision is None
            or (
                bool(runtimes)
                and all(
                    runtime.get("weight_execution_precision")
                    == required_inference_precision
                    for runtime in runtimes
                )
            )
        ),
        "no_final_test_records": (
            bundle["manifest"].get("final_test_record_count") == 0
        ),
        "not_installable_teacher_vault": (
            bundle["manifest"].get("installable_as_layercake_cake") is False
            and bundle["manifest"].get("contains_teacher_material") is True
        ),
        "minimum_functional_passes_met": (
            minimum_functional_passes is None
            or summary["passing_probe_results"] >= minimum_functional_passes
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    evidence: dict[str, Any] = {
        "format": "abi-source-runtime-evidence-audit/1",
        "status": (
            "PASS_SOURCE_RUNTIME_EVIDENCE_PREFLIGHT"
            if not failures
            else "FAIL_SOURCE_RUNTIME_EVIDENCE_PREFLIGHT"
        ),
        "bundle": {
            "path": str(bundle_path),
            "sha256": _sha256_file(bundle_path),
            "manifest_sha256": bundle["verification"]["manifest_sha256"],
            "source_manifest_sha256": sorted(
                str(source["source_manifest_sha256"])
                for source in bundle["sources"]
            ),
        },
        "summary": summary,
        "source_accounting": {
            key: ledger.get(key)
            for key in (
                "teacher_tokens",
                "teacher_generated_output_bytes",
                "raw_source_prompt_count",
                "source_model_inference_seconds",
                "one_time_source_extraction_seconds",
                "source_extraction_devices",
                "source_inference_runtimes",
                "source_parameter_count_read",
                "source_weight_bytes_read",
            )
        },
        "checks": checks,
        "failures": failures,
        "functional_results_are_diagnostic_not_promotion_evidence": True,
        "minimum_functional_passes": minimum_functional_passes,
        "layercake_training_authorized": False,
        "final_test_accessed": False,
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--allow-non-cuda", action="store_true")
    parser.add_argument("--minimum-functional-passes", type=int)
    parser.add_argument("--required-inference-precision")
    args = parser.parse_args(argv)
    evidence = audit_source_runtime_evidence(
        bundle_path=args.bundle,
        output_path=args.output,
        expected_records=args.expected_records,
        require_cuda=not args.allow_non_cuda,
        minimum_functional_passes=args.minimum_functional_passes,
        required_inference_precision=args.required_inference_precision,
    )
    print(json.dumps(evidence, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if not evidence["failures"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
