"""Fail-closed recomputation of the one-pass R50-v1a Phi source capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from abi.capability_compiler_phase2_common import sha256_file
from abi.capability_compiler_phase2_prepare import (
    SOURCE_MANIFEST_SHA256,
    SOURCE_MODEL,
    SOURCE_REVISION,
    _tokenizer,
    _verified_snapshot,
)
from abi.english_generalization_evaluation import _collapse_metrics
from abi.hf_extraction import evaluate_output, load_probe_catalog
from experiments.external_router_replication_r50.acquire_source_v1a import (
    CATALOG_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_ROWS,
)


class SourceVerificationError(RuntimeError):
    pass


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise SourceVerificationError("source raw rows are missing")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SourceVerificationError(f"invalid source JSONL line {line_number}") from error
            if not isinstance(row, dict):
                raise SourceVerificationError(f"source row {line_number} is not an object")
            rows.append(row)
    return rows


def verify(root: Path, catalog_path: Path, evidence: Path) -> dict[str, Any]:
    receipt_path = evidence / "receipt.json"
    rows_path = evidence / "source_outputs.jsonl"
    if not receipt_path.is_file():
        raise SourceVerificationError("source receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256_file(catalog_path) != CATALOG_SHA256:
        raise SourceVerificationError("source catalog hash changed")
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    catalog = load_probe_catalog(catalog_path)
    probes = [dict(row) for row in catalog["probes"] if row["split"] == "final_test"]
    rows = _load_jsonl(rows_path)
    if len(probes) != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS:
        raise SourceVerificationError("source matrix depth changed")
    if receipt.get("outputs") != {
        "path": rows_path.name,
        "sha256": sha256_file(rows_path),
        "bytes": rows_path.stat().st_size,
    }:
        raise SourceVerificationError("source raw-row receipt binding changed")

    expected_keys = {
        "catalog_index", "probe_id", "capability", "split", "prompt",
        "prompt_sha256", "rendered_prompt_sha256", "input_tokens",
        "max_new_tokens", "output", "output_sha256", "output_token_ids",
        "teacher_tokens", "teacher_token_count_authoritative", "evaluator",
        "passed", "score", "collapse", "generation_batch_id",
        "batch_sequences", "batch_latency_seconds",
    }
    for index, (probe, row) in enumerate(zip(probes, rows)):
        if set(row) != expected_keys:
            raise SourceVerificationError(f"source row {index} schema changed")
        prompt = str(probe["prompt"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_tokens = len(tokenizer(rendered, add_special_tokens=False).input_ids)
        token_ids = row["output_token_ids"]
        if (
            row["catalog_index"] != index
            or row["probe_id"] != probe["probe_id"]
            or row["capability"] != probe["capability"]
            or row["split"] != "final_test"
            or row["prompt"] != prompt
            or row["prompt_sha256"] != _digest(prompt)
            or row["rendered_prompt_sha256"] != _digest(rendered)
            or row["input_tokens"] != input_tokens
            or row["max_new_tokens"] != int(probe["max_new_tokens"])
            or row["evaluator"] != probe["evaluator"]
            or not isinstance(token_ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
            or len(token_ids) > int(probe["max_new_tokens"])
            or row["teacher_tokens"] != len(token_ids)
            or row["teacher_token_count_authoritative"] is not True
        ):
            raise SourceVerificationError(f"source row {index} identity changed")
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        if row["output"] != decoded or row["output_sha256"] != _digest(decoded):
            raise SourceVerificationError(f"source row {index} token/text binding changed")
        passed, score = evaluate_output(decoded, probe["evaluator"])
        collapse = _collapse_metrics(token_ids, decoded, tokenizer.encode(prompt + "\n"), prompt)
        if row["passed"] is not passed or row["score"] != score or row["collapse"] != collapse:
            raise SourceVerificationError(f"source row {index} evaluation changed")
        if (
            isinstance(row["generation_batch_id"], bool)
            or not isinstance(row["generation_batch_id"], int)
            or row["generation_batch_id"] < 0
            or not isinstance(row["batch_sequences"], int)
            or not 1 <= row["batch_sequences"] <= 8
            or not isinstance(row["batch_latency_seconds"], (int, float))
            or not math.isfinite(row["batch_latency_seconds"])
            or row["batch_latency_seconds"] <= 0
        ):
            raise SourceVerificationError(f"source row {index} timing metadata changed")

    batches: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        batches.setdefault(int(row["generation_batch_id"]), []).append(row)
    if sorted(batches) != list(range(len(batches))):
        raise SourceVerificationError("source batch IDs are not contiguous")
    for batch_id, values in batches.items():
        if (
            len(values) != values[0]["batch_sequences"]
            or any(row["batch_sequences"] != len(values) for row in values)
            or len({row["max_new_tokens"] for row in values}) != 1
            or len({row["batch_latency_seconds"] for row in values}) != 1
        ):
            raise SourceVerificationError(f"source batch {batch_id} metadata changed")

    counts = dict(sorted(Counter(row["capability"] for row in rows).items()))
    recomputed = {
        "observations": len(rows),
        "capability_counts": counts,
        "functional_passes": sum(row["passed"] for row in rows),
        "repetition_collapses": sum(row["collapse"]["collapse_detected"] for row in rows),
        "teacher_tokens": sum(row["teacher_tokens"] for row in rows),
        "output_utf8_bytes": sum(len(row["output"].encode("utf-8")) for row in rows),
    }
    for key, value in recomputed.items():
        if receipt.get(key) != value:
            raise SourceVerificationError(f"source receipt aggregate changed: {key}")
    generation = receipt.get("generation", {})
    if (
        receipt.get("format") != "abi-r50-live-source-capture/1"
        or receipt.get("source_model") != SOURCE_MODEL
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or receipt.get("source_parameters") != EXPECTED_PARAMETERS
        or receipt.get("catalog_sha256") != CATALOG_SHA256
        or receipt.get("split") != "final_test"
        or generation.get("do_sample") is not False
        or generation.get("generation_calls") != len(batches)
        or generation.get("sequences") != EXPECTED_ROWS
        or generation.get("batch_size_maximum") != 8
        or receipt.get("training_steps") != 0
        or receipt.get("candidate_accessed") is not False
        or receipt.get("candidate_generations") != 0
        or receipt.get("source_teacher_present_in_final_layercake") is not False
    ):
        raise SourceVerificationError("source receipt contract changed")
    for timing in ("source_load_seconds", "source_inference_seconds", "total_wall_seconds"):
        value = receipt.get(timing)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SourceVerificationError(f"source receipt timing invalid: {timing}")
    result = {
        "verdict": "PASS_STRICT_SOURCE_CAPTURE",
        "raw_rows": len(rows),
        "functional_passes": recomputed["functional_passes"],
        "teacher_tokens": recomputed["teacher_tokens"],
        "collapses": recomputed["repetition_collapses"],
        "batches": len(batches),
        "outputs_sha256": sha256_file(rows_path),
        "receipt_sha256": sha256_file(receipt_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    verify(Path.cwd().resolve(), args.catalog.resolve(), args.evidence.resolve())


if __name__ == "__main__":
    main()
