"""Fail-closed recomputation of the R60 live source capture."""

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
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once
from experiments.prospective_endpoint_r60.acquire_source_v1 import (
    CATALOG_SHA256,
    EXPECTED_PARAMETERS,
    EXPECTED_ROWS,
)


class SourceVerificationError(RuntimeError):
    pass


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.stat().st_size:
        raise SourceVerificationError("source raw rows are absent")
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceVerificationError("source raw rows are unreadable") from error
    if any(not isinstance(row, dict) for row in rows):
        raise SourceVerificationError("source raw rows contain a non-object")
    return rows


def verify(root: Path, catalog_path: Path, evidence: Path, output: Path) -> dict[str, Any]:
    receipt_path = evidence / "receipt.json"
    rows_path = evidence / "source_outputs.jsonl"
    if output.exists():
        raise SourceVerificationError(f"immutable verification exists: {output}")
    if not receipt_path.is_file():
        raise SourceVerificationError("source receipt is absent")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if sha256_file(catalog_path) != CATALOG_SHA256:
        raise SourceVerificationError("prospective catalog changed")
    snapshot = _verified_snapshot(root)
    tokenizer = _tokenizer(snapshot)
    probes = list(load_probe_catalog(catalog_path)["probes"])
    rows = _jsonl(rows_path)
    if len(probes) != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS:
        raise SourceVerificationError("source matrix depth changed")
    if receipt.get("outputs") != {
        "path": rows_path.name,
        "sha256": sha256_file(rows_path),
        "bytes": rows_path.stat().st_size,
    }:
        raise SourceVerificationError("source raw-row binding changed")

    expected_keys = {
        "catalog_index", "probe_id", "capability", "split", "prompt",
        "prompt_sha256", "rendered_prompt_sha256", "input_tokens",
        "max_new_tokens", "output", "output_sha256", "output_token_ids",
        "teacher_tokens", "teacher_token_count_authoritative", "evaluator",
        "passed", "score", "collapse", "generation_batch_id",
        "batch_sequences", "batch_latency_seconds",
    }
    for index, (probe, row) in enumerate(zip(probes, rows, strict=True)):
        if set(row) != expected_keys:
            raise SourceVerificationError(f"source row {index} schema changed")
        prompt = str(probe["prompt"])
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = row["output_token_ids"]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        passed, score = evaluate_output(decoded, probe["evaluator"])
        collapse = _collapse_metrics(
            token_ids, decoded, tokenizer.encode(prompt + "\n"), prompt
        )
        if (
            row["catalog_index"] != index
            or row["probe_id"] != probe["probe_id"]
            or row["capability"] != probe["capability"]
            or row["split"] != "prospective_r60"
            or row["prompt"] != prompt
            or row["prompt_sha256"] != _digest(prompt)
            or row["rendered_prompt_sha256"] != _digest(rendered)
            or row["input_tokens"]
            != len(tokenizer(rendered, add_special_tokens=False).input_ids)
            or row["max_new_tokens"] != int(probe["max_new_tokens"])
            or row["evaluator"] != probe["evaluator"]
            or not isinstance(token_ids, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
            or len(token_ids) > int(probe["max_new_tokens"])
            or row["teacher_tokens"] != len(token_ids)
            or row["teacher_token_count_authoritative"] is not True
            or row["output"] != decoded
            or row["output_sha256"] != _digest(decoded)
            or row["passed"] is not bool(passed)
            or row["score"] != float(score)
            or row["collapse"] != collapse
            or not isinstance(row["batch_latency_seconds"], (int, float))
            or not math.isfinite(row["batch_latency_seconds"])
            or row["batch_latency_seconds"] <= 0
        ):
            raise SourceVerificationError(f"source row {index} changed")

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
            raise SourceVerificationError(f"source batch {batch_id} changed")

    aggregate = {
        "observations": len(rows),
        "capability_counts": dict(
            sorted(Counter(row["capability"] for row in rows).items())
        ),
        "functional_passes": sum(row["passed"] for row in rows),
        "repetition_collapses": sum(
            row["collapse"]["collapse_detected"] for row in rows
        ),
        "teacher_tokens": sum(row["teacher_tokens"] for row in rows),
        "output_utf8_bytes": sum(len(row["output"].encode("utf-8")) for row in rows),
    }
    if any(receipt.get(key) != value for key, value in aggregate.items()):
        raise SourceVerificationError("source receipt aggregate changed")
    generation = receipt.get("generation", {})
    if (
        receipt.get("format") != "abi-r60-live-source-capture/1"
        or receipt.get("status") != "COMPLETE_PROSPECTIVE_SOURCE_CAPTURE"
        or receipt.get("source_model") != SOURCE_MODEL
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA256
        or receipt.get("source_parameters") != EXPECTED_PARAMETERS
        or receipt.get("catalog_sha256") != CATALOG_SHA256
        or receipt.get("split") != "prospective_r60"
        or generation.get("do_sample") is not False
        or generation.get("generation_calls") != len(batches)
        or generation.get("sequences") != EXPECTED_ROWS
        or generation.get("batch_size_maximum") != 8
        or receipt.get("training_steps") != 0
        or receipt.get("candidate_accessed") is not False
        or receipt.get("candidate_generations") != 0
        or receipt.get("source_teacher_present_in_final_layercake") is not False
        or receipt.get("full_abi_moonshot") != "OPEN"
    ):
        raise SourceVerificationError("source receipt contract changed")
    for key in ("source_load_seconds", "source_inference_seconds", "total_wall_seconds"):
        value = receipt.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise SourceVerificationError(f"source timing invalid: {key}")

    result = {
        "format": "abi-r60-strict-source-verification/1",
        "verdict": "PASS_R60_STRICT_SOURCE_CAPTURE",
        "raw_rows_recomputed": len(rows),
        "functional_passes": aggregate["functional_passes"],
        "teacher_tokens": aggregate["teacher_tokens"],
        "collapses": aggregate["repetition_collapses"],
        "batches": len(batches),
        "outputs_sha256": sha256_file(rows_path),
        "receipt_sha256": sha256_file(receipt_path),
        "source_snapshot_rehashed": True,
        "candidate_loaded": False,
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(Path.cwd(), args.catalog.resolve(), args.evidence.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()

