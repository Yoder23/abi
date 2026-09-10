"""Acquire R21 teacher labels and responses on the registered GPU."""

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
    evidence_hash,
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

from .binding import load_config
from .protocol import LABELS, score_output, training_rows

OUTPUT_SYSTEM = (
    "Follow the instruction using only the three supplied DATA values. Preserve "
    "each value, add no facts, and return only the requested output."
)
LABEL_SYSTEM = (
    "Classify the requested behavior. Return exactly one lowercase label and no "
    "other text: prose, summary, email, bullets, clarification, or abstention."
)


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 source output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_config(root, config_path)
    source = config["source"]
    if not torch.cuda.is_available():
        raise R14Error("R21 registered CUDA source acquisition unavailable")
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    cpu_started = time.process_time()
    load_started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(str(source["model_id"]), str(source["revision"]))
    load_seconds = time.perf_counter() - load_started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    observations = []
    counts = {
        "raw_source_prompts": 0,
        "raw_source_prompt_utf8_bytes": 0,
        "rendered_prompt_utf8_bytes": 0,
        "rendered_prompt_token_instances": 0,
        "teacher_generated_tokens": 0,
        "teacher_output_bytes": 0,
        "teacher_label_tokens": 0,
        "teacher_label_bytes": 0,
    }
    inference_started = time.perf_counter()
    rows = training_rows()
    for index, row in enumerate(rows, 1):
        prompt = str(row["prompt"])
        output_rendered = _render_chat(tokenizer, OUTPUT_SYSTEM, prompt)
        response, response_tokens = _generate(
            tokenizer, model, output_rendered, int(source["max_new_tokens"])
        )
        response = response.strip().replace("\r\n", "\n")
        label_rendered = _render_chat(tokenizer, LABEL_SYSTEM, prompt)
        label_raw, label_tokens = _generate(
            tokenizer, model, label_rendered, int(source["max_label_tokens"])
        )
        label = label_raw.strip().lower().rstrip(".")
        input_tokens = len(tokenizer.encode(output_rendered, add_special_tokens=False))
        label_input_tokens = len(tokenizer.encode(label_rendered, add_special_tokens=False))
        scored = score_output(row, response, response)
        observations.append(
            {
                **row,
                "teacher_output": response,
                "teacher_output_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "teacher_output_token_count": response_tokens,
                "teacher_label": label,
                "teacher_label_raw": label_raw.strip(),
                "teacher_label_token_count": label_tokens,
                "teacher_label_valid": label in LABELS,
                "teacher_label_exact": label == row["task"],
                "teacher_quality": scored,
                "rendered_output_prompt_sha256": hashlib.sha256(output_rendered.encode()).hexdigest(),
                "rendered_label_prompt_sha256": hashlib.sha256(label_rendered.encode()).hexdigest(),
                "output_input_token_count": input_tokens,
                "label_input_token_count": label_input_tokens,
            }
        )
        counts["raw_source_prompts"] += 2
        counts["raw_source_prompt_utf8_bytes"] += 2 * len(prompt.encode())
        counts["rendered_prompt_utf8_bytes"] += len(output_rendered.encode()) + len(label_rendered.encode())
        counts["rendered_prompt_token_instances"] += input_tokens + label_input_tokens
        counts["teacher_generated_tokens"] += response_tokens
        counts["teacher_output_bytes"] += len(response.encode())
        counts["teacher_label_tokens"] += label_tokens
        counts["teacher_label_bytes"] += len(label_raw.strip().encode())
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "rows": index,
                        "label_exact": sum(item["teacher_label_exact"] for item in observations),
                        "functional": sum(item["teacher_quality"]["functional_pass"] for item in observations),
                        "seconds": time.perf_counter() - inference_started,
                    }
                ),
                flush=True,
            )
    inference_seconds = time.perf_counter() - inference_started
    peak_gpu = int(torch.cuda.max_memory_allocated())
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "source_observations.jsonl"
    write_jsonl_once(rows_path, observations)
    snapshot_files = [path for path in snapshot.rglob("*") if path.is_file()]
    label_exact = sum(row["teacher_label_exact"] for row in observations)
    label_valid = sum(row["teacher_label_valid"] for row in observations)
    receipt = {
        "format": "abi-r21-public-source-acquisition/1",
        "verdict": "PASS_SOURCE" if label_exact >= 594 and label_valid == 600 else "FAIL_SOURCE",
        "config_sha256": sha256_file(config_path),
        "source": {
            "model_id": source["model_id"],
            "revision": source["revision"],
            "parameters": parameters,
            "device": "cuda",
            "training_steps": 0,
            "present_at_student_training": False,
            "present_at_package_execution": False,
        },
        "metrics": {
            "rows": len(observations),
            "valid_labels": label_valid,
            "exact_labels": label_exact,
            "teacher_functional": sum(row["teacher_quality"]["functional_pass"] for row in observations),
            "teacher_non_hallucinating": sum(row["teacher_quality"]["hallucination_pass"] for row in observations),
            "teacher_non_collapsed": sum(not row["teacher_quality"]["repetition_collapse"] for row in observations),
        },
        "artifacts": {"source_rows": {"path": rows_path.name, "sha256": sha256_file(rows_path)}},
        "information_accounting": {
            **counts,
            "unique_source_prompt_utf8_bytes": sum(len(str(row["prompt"]).encode()) for row in rows),
            "unique_teacher_output_utf8_bytes": sum(len(value.encode()) for value in {row["teacher_output"] for row in observations}),
            "unique_teacher_label_utf8_bytes": sum(len(value.encode()) for value in {row["teacher_label"] for row in observations}),
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_process_seconds": time.process_time() - cpu_started,
            "source_load_seconds": load_seconds,
            "source_inference_seconds": inference_seconds,
            "source_model_inference_hours": inference_seconds / 3600,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_cpu_rss_bytes_observed": peak_rss,
            "gpu_name": gpu_name,
            "external_hardware_used": False,
            "source_parameters": parameters,
            "source_snapshot_files": len(snapshot_files),
            "source_snapshot_disk_bytes": sum(path.stat().st_size for path in snapshot_files),
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "frozen_source_parameters_copied": 0,
        },
        "full_abi_moonshot": "OPEN",
    }
    receipt["evidence_sha256"] = evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
