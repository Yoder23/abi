"""Normalize preserved R21 teacher responses into R22 semantic-plan targets."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch

from experiments.factual_semantic_r16.public_qualification import (
    _generate,
    _load_source,
    _render_chat,
)
from experiments.foreign_capability_r14.core import (
    R14Error,
    json_object,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)
from experiments.generative_transfer_r21.hash_assurance_binding import (
    selfless_evidence_hash,
)
from experiments.generative_transfer_r21.protocol import score_output, training_rows

from .binding import load_config
from .protocol import (
    NORMALIZATION_SYSTEM,
    fields_verbatim_once,
    normalization_prompt,
    normalization_request_sha256,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error(f"R22 JSONL unreadable: {path}") from exc


def validate_source(
    root: Path, config_path: Path, source_run: Path
) -> list[dict[str, Any]]:
    config = load_config(root, config_path)
    receipt = json_object(source_run / "receipt.json")
    rows_path = source_run / "source_observations.jsonl"
    rows = _jsonl(rows_path)
    original = _jsonl(root / str(config["raw_source_rows"]["path"]))
    expected = training_rows()
    scores = [
        score_output(
            reference,
            str(row.get("teacher_output", "")),
            str(row.get("raw_teacher_output", "")),
        )
        for row, reference in zip(rows, expected)
    ]
    verbatim = [
        fields_verbatim_once(reference, str(row.get("teacher_output", "")))
        for row, reference in zip(rows, expected)
    ]
    gates = {
        "functional": sum(row["functional_pass"] for row in scores)
        >= int(config["gates"]["minimum_functional"]),
        "non_hallucinating": sum(row["hallucination_pass"] for row in scores)
        >= int(config["gates"]["minimum_non_hallucinating"]),
        "non_collapsed": sum(not row["repetition_collapse"] for row in scores)
        >= int(config["gates"]["minimum_non_collapsed"]),
        "fields_verbatim_once": sum(verbatim)
        == int(config["gates"]["required_fields_verbatim_once"]),
        "label_exact": sum(
            row.get("teacher_label") == reference["task"]
            for row, reference in zip(rows, expected)
        )
        == int(config["gates"]["required_label_exact"]),
    }
    metrics = {
        "rows": len(rows),
        "functional": sum(row["functional_pass"] for row in scores),
        "non_hallucinating": sum(row["hallucination_pass"] for row in scores),
        "non_collapsed": sum(not row["repetition_collapse"] for row in scores),
        "fields_verbatim_once": sum(verbatim),
    }
    original_identity = [
        (row.get("record_id"), row.get("teacher_output"), row.get("teacher_output_sha256"))
        for row in original
    ]
    carried_identity = [
        (row.get("record_id"), row.get("raw_teacher_output"), row.get("raw_teacher_output_sha256"))
        for row in rows
    ]
    if (
        receipt.get("format") != "abi-r22-normalized-source/1"
        or receipt.get("verdict")
        != ("PASS_NORMALIZED_SOURCE" if all(gates.values()) else "FAIL_NORMALIZED_SOURCE")
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("evidence_sha256") != selfless_evidence_hash(receipt)
        or receipt.get("source_rows_sha256") != sha256_file(rows_path)
        or len(rows) != 600
        or len(original) != 600
        or [row.get("record_id") for row in rows]
        != [row["record_id"] for row in expected]
        or original_identity != carried_identity
        or [row.get("normalized_quality") for row in rows] != scores
        or [row.get("fields_verbatim_once") for row in rows] != verbatim
        or any(
            row.get("teacher_output_sha256")
            != hashlib.sha256(str(row.get("teacher_output", "")).encode()).hexdigest()
            for row in rows
        )
        or any(
            row.get("normalization_request_sha256")
            != normalization_request_sha256(original_row)
            for row, original_row in zip(rows, original)
        )
        or receipt.get("metrics") != metrics
        or receipt.get("gates") != gates
        or receipt.get("source", {}).get("model_id") != config["source"]["model_id"]
        or receipt.get("source", {}).get("revision") != config["source"]["revision"]
        or receipt.get("source", {}).get("calls") != 600
        or receipt.get("source", {}).get("training_steps") != 0
        or receipt.get("source", {}).get("present_at_student_training") is not False
        or receipt.get("source", {}).get("present_at_package_execution") is not False
        or receipt.get("information_accounting", {}).get("normalization_calls") != 600
        or receipt.get("information_accounting", {}).get("logits_stored") != 0
        or receipt.get("information_accounting", {}).get("hidden_activations_stored")
        != 0
        or receipt.get("information_accounting", {}).get("source_parameters_copied")
        != 0
    ):
        raise R14Error("R22 normalized source evidence failed")
    return rows


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R22 normalized source exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_config(root, config_path)
    original = _jsonl(root / str(config["raw_source_rows"]["path"]))
    expected = training_rows()
    if (
        len(original) != 600
        or [row.get("record_id") for row in original]
        != [row["record_id"] for row in expected]
        or any(row.get("teacher_label") != reference["task"] for row, reference in zip(original, expected))
    ):
        raise R14Error("R22 raw source prerequisite changed")
    if not torch.cuda.is_available():
        raise R14Error("R22 normalization requires CUDA")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(
        config["source"]["model_id"], config["source"]["revision"]
    )
    load_seconds = time.perf_counter() - started
    observations = []
    prompt_tokens = 0
    output_tokens = 0
    output_bytes = 0
    inference_started = time.perf_counter()
    for index, (row, reference) in enumerate(zip(original, expected), 1):
        transfer_prompt = normalization_prompt(row)
        rendered = _render_chat(tokenizer, NORMALIZATION_SYSTEM, transfer_prompt)
        normalized, tokens = _generate(
            tokenizer, model, rendered, int(config["source"]["max_new_tokens"])
        )
        normalized = normalized.strip().replace("\r\n", "\n")
        score = score_output(reference, normalized, row["teacher_output"])
        verbatim = fields_verbatim_once(reference, normalized)
        observations.append(
            {
                **row,
                "raw_teacher_output": row["teacher_output"],
                "raw_teacher_output_sha256": row["teacher_output_sha256"],
                "teacher_output": normalized,
                "teacher_output_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                "teacher_output_token_count": tokens,
                "teacher_label_token_count": 0,
                "normalized_quality": score,
                "fields_verbatim_once": verbatim,
                "normalization_request_sha256": normalization_request_sha256(row),
            }
        )
        prompt_tokens += len(tokenizer.encode(rendered, add_special_tokens=False))
        output_tokens += tokens
        output_bytes += len(normalized.encode())
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if index == 1 or index % 25 == 0 or index == len(original):
            print(
                json.dumps(
                    {
                        "normalized_rows": index,
                        "functional": sum(
                            item["normalized_quality"]["functional_pass"]
                            for item in observations
                        ),
                        "non_hallucinating": sum(
                            item["normalized_quality"]["hallucination_pass"]
                            for item in observations
                        ),
                    }
                ),
                flush=True,
            )
    inference_seconds = time.perf_counter() - inference_started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    peak_gpu = int(torch.cuda.max_memory_allocated())
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "source_observations.jsonl"
    write_jsonl_once(rows_path, observations)
    scores = [row["normalized_quality"] for row in observations]
    gates = {
        "functional": sum(row["functional_pass"] for row in scores)
        >= int(config["gates"]["minimum_functional"]),
        "non_hallucinating": sum(row["hallucination_pass"] for row in scores)
        >= int(config["gates"]["minimum_non_hallucinating"]),
        "non_collapsed": sum(not row["repetition_collapse"] for row in scores)
        >= int(config["gates"]["minimum_non_collapsed"]),
        "fields_verbatim_once": sum(
            bool(row["fields_verbatim_once"]) for row in observations
        )
        == int(config["gates"]["required_fields_verbatim_once"]),
        "label_exact": all(
            row["teacher_label"] == reference["task"]
            for row, reference in zip(observations, expected)
        ),
    }
    snapshot_files = [path for path in snapshot.rglob("*") if path.is_file()]
    receipt = {
        "format": "abi-r22-normalized-source/1",
        "verdict": "PASS_NORMALIZED_SOURCE" if all(gates.values()) else "FAIL_NORMALIZED_SOURCE",
        "config_sha256": sha256_file(config_path),
        "raw_source_receipt_sha256": config["raw_source_receipt"]["sha256"],
        "raw_source_rows_sha256": config["raw_source_rows"]["sha256"],
        "source_rows_sha256": sha256_file(rows_path),
        "metrics": {
            "rows": len(observations),
            "functional": sum(row["functional_pass"] for row in scores),
            "non_hallucinating": sum(row["hallucination_pass"] for row in scores),
            "non_collapsed": sum(not row["repetition_collapse"] for row in scores),
            "fields_verbatim_once": sum(
                bool(row["fields_verbatim_once"]) for row in observations
            ),
        },
        "gates": gates,
        "source": {
            "model_id": config["source"]["model_id"],
            "revision": config["source"]["revision"],
            "parameters": parameters,
            "calls": len(observations),
            "training_steps": 0,
            "present_at_student_training": False,
            "present_at_package_execution": False,
        },
        "information_accounting": {
            "raw_teacher_response_tokens_reused": sum(
                int(row.get("teacher_output_token_count", 0)) for row in original
            ),
            "raw_teacher_response_bytes_reused": sum(
                len(row["teacher_output"].encode()) for row in original
            ),
            "normalization_calls": len(observations),
            "normalization_prompt_token_instances": prompt_tokens,
            "normalization_output_tokens": output_tokens,
            "normalization_output_bytes": output_bytes,
            "source_load_seconds": load_seconds,
            "source_inference_seconds": inference_seconds,
            "source_model_inference_hours": inference_seconds / 3600,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_cpu_rss_bytes": peak_rss,
            "gpu_name": gpu_name,
            "source_snapshot_files": len(snapshot_files),
            "source_snapshot_bytes": sum(path.stat().st_size for path in snapshot_files),
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
        },
        "full_abi_moonshot": "OPEN",
        "next_action": "run frozen matched bakeoff" if all(gates.values()) else "close R22 normalization branch",
    }
    receipt["evidence_sha256"] = selfless_evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
