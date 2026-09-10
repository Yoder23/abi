"""Repair R16 accounting with exact rendered source-prompt measurements."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiments.foreign_capability_r14.core import (
    R14Error,
    evidence_hash,
    json_object,
    write_json_once,
)

from .accounting import account
from .public_qualification import _render_chat

SYSTEM_PROMPT = "Answer the factual question with only the answer and no explanation."


def _tokenizer(model_id: str, revision: str) -> Any:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(
        snapshot_download(model_id, revision=revision, local_files_only=True)
    ).resolve()
    if snapshot.name != revision:
        raise R14Error("R16 accounting source revision changed")
    return AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )


def account_v2(run_dir: Path) -> dict[str, Any]:
    receipt = json_object(run_dir / "receipt.json")
    source_rows = [
        json.loads(line)
        for line in (run_dir / receipt["artifacts"]["source_rows"]["path"])
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if len(source_rows) != receipt["information_accounting"]["raw_source_prompts"]:
        raise R14Error("R16 v2 accounting source-row count changed")
    tokenizer = _tokenizer(
        str(receipt["source"]["model_id"]), str(receipt["source"]["revision"])
    )
    rendered = [
        _render_chat(tokenizer, SYSTEM_PROMPT, str(row["question"]))
        for row in source_rows
    ]
    rendered_tokens = [
        tokenizer(value, add_special_tokens=False)["input_ids"] for value in rendered
    ]
    extraction_rows = [row for row in source_rows if row["split"] == "extraction"]
    candidate_strings = [
        str(candidate)
        for row in extraction_rows
        for candidate in row["candidate_values"]
    ]
    candidate_token_ids = [
        tokenizer.encode(value, add_special_tokens=False) for value in candidate_strings
    ]
    if any(not value for value in candidate_token_ids):
        raise R14Error("R16 v2 accounting found an empty candidate token sequence")
    unique_rendered = set(rendered)
    unique_candidates = set(candidate_strings)
    legacy = account(run_dir)
    is_v1_repair = run_dir.name == "heldout_v1"
    result = {
        "format": "abi-r16-imported-information-accounting/2",
        "repair_of": (
            "results/factual_semantic_r16/heldout_v1_accounting.json"
            if is_v1_repair
            else None
        ),
        "measurement_note": (
            "v1 unique_prompt_utf8_bytes counted question text rather than the exact "
            "tokenizer-rendered chat prompts supplied to the source"
            if is_v1_repair
            else "first accounting receipt for this replication uses exact rendered prompts"
        ),
        "claim_ceiling": "R16_BOUNDED_FACTUAL_ACQUISITION_ONLY",
        "raw_source_prompt_instances": len(rendered),
        "unique_source_prompts": len(unique_rendered),
        "raw_source_prompt_utf8_bytes": sum(len(value.encode()) for value in rendered),
        "unique_source_prompt_utf8_bytes": sum(
            len(value.encode()) for value in unique_rendered
        ),
        "raw_question_utf8_bytes": sum(
            len(str(row["question"]).encode()) for row in source_rows
        ),
        "unique_question_utf8_bytes": sum(
            len(value.encode()) for value in {str(row["question"]) for row in source_rows}
        ),
        "source_prompt_token_instances": sum(len(value) for value in rendered_tokens),
        "teacher_generated_tokens": legacy["teacher_generated_tokens"],
        "teacher_generated_output_bytes": legacy["teacher_generated_output_bytes"],
        "candidate_string_instances": len(candidate_strings),
        "unique_candidate_strings": len(unique_candidates),
        "candidate_string_utf8_bytes": sum(
            len(value.encode()) for value in candidate_strings
        ),
        "unique_candidate_string_utf8_bytes": sum(
            len(value.encode()) for value in unique_candidates
        ),
        "candidate_token_instances_scored": sum(
            len(value) for value in candidate_token_ids
        ),
        "candidate_score_values_imported": legacy["candidate_score_values_imported"],
        "hidden_activation_values_stored_for_evidence": legacy[
            "hidden_activation_values_stored_for_evidence"
        ],
        "hidden_activation_bytes_stored_for_evidence": legacy[
            "hidden_activation_bytes_stored_for_evidence"
        ],
        "hidden_activations_consumed_by_compiler": 0,
        "source_bundle_disk_bytes": legacy["source_bundle_disk_bytes"],
        "source_residual_evidence_disk_bytes": legacy[
            "source_residual_evidence_disk_bytes"
        ],
        "final_package_disk_bytes": legacy["final_package_disk_bytes"],
        "final_structured_fact_records": legacy["final_structured_fact_records"],
        "frozen_source_parameters_in_final_packages": 0,
        "bridge_parameters_trained": 0,
        "source_training_steps": 0,
        "end_to_end_acquisition_seconds": legacy["end_to_end_acquisition_seconds"],
        "source_inference_seconds": "NOT_SEPARATELY_MEASURED",
        "peak_gpu_memory_bytes": legacy["peak_gpu_memory_bytes"],
        "peak_cpu_ram_bytes": "NOT_MEASURED",
        "external_hardware_used": False,
        "gpu_used": True,
        "superseded_v1_evidence_sha256": legacy["evidence_sha256"],
    }
    if result["candidate_string_instances"] != result["candidate_score_values_imported"]:
        raise R14Error("R16 candidate accounting is not one-to-one")
    result["evidence_sha256"] = evidence_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = account_v2(args.run_dir)
    write_json_once(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
