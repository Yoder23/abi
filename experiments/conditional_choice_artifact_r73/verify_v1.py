"""Fail-closed independent verifier for the frozen R73 capability artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from abi.capability_pipeline import read_extraction_bundle
from abi.conditional_choice_source_artifact import (
    COUNTER,
    EXPECTED_CATALOG_SHA256,
    EXPECTED_EVIDENCE_SHA256,
    EXPECTED_PASSING,
    EXPECTED_RAW_SHA256,
    EXPECTED_RESULT_FILE_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    _canonical_sha,
    _file_sha,
    _load_jsonl,
    _load_source_token_counter,
    _validate_result,
    _validated_rows,
)
from abi.hf_extraction import load_probe_catalog
from abi.layercake_host_v3 import LayerCakeHostError, _materialize_training_prompt
from experiments.foreign_capability_r14.core import write_json_once


EXPECTED_ARCHIVE_SHA256 = "0af70b5bc4812c7a957554b14054f8561875c94977d5555bc8a899716a8dbf97"
EXPECTED_RECEIPT_FILE_SHA256 = "c8e099316c172391d124a4909580789728dae5f39c92af1358d8a8dba1b776d9"


class VerificationError(RuntimeError):
    pass


def verify(
    *,
    artifact_path: Path,
    receipt_path: Path,
    catalog_path: Path,
    result_path: Path,
    raw_scores_path: Path,
) -> dict:
    if _file_sha(artifact_path) != EXPECTED_ARCHIVE_SHA256:
        raise VerificationError("R73 archive identity changed")
    if _file_sha(receipt_path) != EXPECTED_RECEIPT_FILE_SHA256:
        raise VerificationError("R73 receipt identity changed")
    if _file_sha(catalog_path) != EXPECTED_CATALOG_SHA256:
        raise VerificationError("R72 catalog identity changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _validate_result(result, result_path, raw_scores_path)
    catalog = load_probe_catalog(catalog_path)
    raw_rows = _load_jsonl(raw_scores_path)
    passing, per_family = _validated_rows(catalog=catalog, raw_rows=raw_rows)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        receipt.get("receipt_sha256") != _canonical_sha(receipt_payload)
        or receipt.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256
        or receipt.get("records") != EXPECTED_PASSING
        or receipt.get("verified") is not True
    ):
        raise VerificationError("R73 receipt is stale")

    bundle = read_extraction_bundle(artifact_path)
    if (
        bundle["verification"]["verified"] is not True
        or bundle["verification"]["archive_sha256"] != EXPECTED_ARCHIVE_SHA256
        or bundle["verification"]["record_count"] != EXPECTED_PASSING
        or bundle["verification"]["domain_segregation_verified"] is not True
        or len(bundle["sources"]) != 1
        or bundle["sources"][0]["source_manifest_sha256"]
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise VerificationError("R73 bundle boundary changed")
    source = bundle["sources"][0]
    token_counter = _load_source_token_counter(
        model_id=source["model_id"],
        revision=source["revision"],
        trust_remote_code=bool(source["trust_remote_code"]),
    )
    observations = {
        str(probe["probe_id"]): (probe, observation, observation_hash)
        for probe, observation, observation_hash in passing
    }
    records = {str(row["record_id"]): row for row in bundle["records"]}
    results = {str(row["record_id"]): row for row in bundle["probe_results"]}
    if set(records) != set(results) or len(records) != EXPECTED_PASSING:
        raise VerificationError("record/result identity coverage changed")
    total_tokens = 0
    for record_id, record in records.items():
        probe_result = results[record_id]
        probe_id = str(probe_result["probe_id"])
        if probe_id not in observations:
            raise VerificationError("packaged record lacks passing source evidence")
        probe, observation, observation_hash = observations[probe_id]
        evaluator = probe_result["evaluator"]
        if (
            record["teacher_token_counter"] != COUNTER
            or record["prompt"] != probe["prompt"]
            or record["output"] != observation["selected_output"]
            or record["teacher_tokens"] != token_counter(record["output"])
            or record["provenance"]
            != f"conditional-choice:{EXPECTED_EVIDENCE_SHA256}:{observation_hash}"
            or evaluator["conditional_choice_observation_sha256"] != observation_hash
            or evaluator["conditional_choice_evidence_sha256"]
            != EXPECTED_EVIDENCE_SHA256
            or _materialize_training_prompt(
                record=record, probe_result=probe_result, ledger=bundle["ledger"]
            )
            != probe["prompt"]
        ):
            raise VerificationError("packaged row changed its source evidence")
        total_tokens += int(record["teacher_tokens"])
    ledger = bundle["ledger"]
    if (
        ledger["teacher_tokens"] != total_tokens
        or ledger["teacher_generated_output_bytes"] != 0
        or ledger["logits_stored_count"] != 6_300
        or ledger["logits_stored_bytes"] != 50_400
        or ledger["conditional_choice_qualification"]["raw_scores_sha256"]
        != EXPECTED_RAW_SHA256
        or ledger["conditional_choice_qualification"]["evidence_file_sha256"]
        != EXPECTED_RESULT_FILE_SHA256
        or ledger["conditional_choice_qualification"]["family_passing"]
        != per_family
    ):
        raise VerificationError("R73 imported-information accounting is stale")

    first_record = next(iter(records.values()))
    first_result = copy.deepcopy(results[first_record["record_id"]])
    first_result["evaluator"]["top_margin_mean_log_probability"] = 0.0
    hostile_rejected = False
    try:
        _materialize_training_prompt(
            record=first_record, probe_result=first_result, ledger=ledger
        )
    except LayerCakeHostError:
        hostile_rejected = True
    if not hostile_rejected:
        raise VerificationError("host consumer accepted a zero-margin mutation")
    return {
        "format": "abi-conditional-choice-source-artifact-verification/1",
        "verdict": "PASS_R73_EXACT_CONDITIONAL_CAPABILITY_ARTIFACT",
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "receipt_file_sha256": EXPECTED_RECEIPT_FILE_SHA256,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "catalog_sha256": EXPECTED_CATALOG_SHA256,
        "source_evidence_sha256": EXPECTED_EVIDENCE_SHA256,
        "raw_scores_sha256": EXPECTED_RAW_SHA256,
        "source_rows": 2_100,
        "packaged_passing_rows": len(records),
        "teacher_generated_rows": 0,
        "conditional_scores_stored": 6_300,
        "authoritative_selected_output_tokens": total_tokens,
        "family_passing": per_family,
        "host_consumer_rows_revalidated": len(records),
        "hostile_zero_margin_mutation_rejected": hostile_rejected,
        "layercake_training_performed": False,
        "host_certified": False,
        "full_abi_moonshot": "OPEN",
        "claim_boundary": (
            "Exact verification of one bounded, source-weight-selected reasoning "
            "artifact. This is not a general-English or deployed-host result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--raw-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = verify(
        artifact_path=args.artifact,
        receipt_path=args.receipt,
        catalog_path=args.catalog,
        result_path=args.result,
        raw_scores_path=args.raw_scores,
    )
    value["evidence_sha256"] = _canonical_sha(value)
    write_json_once(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
