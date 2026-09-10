"""Acquire disclosed R20 teacher realizations on the declared GPU."""

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

from .binding import SOURCE, load_config
from .compiler import R20CompilerError, infer_package
from .protocol import TASKS, public_rows

SYSTEM = (
    "Follow the instruction using only the three supplied DATA values. Preserve "
    "each value exactly, add no facts, and return only the requested output."
)
MODEL_ID = str(SOURCE["model_id"])
REVISION = str(SOURCE["revision"])
MAX_NEW_TOKENS = int(SOURCE["max_new_tokens"])


def _compiler_records(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "record_id": str(row["record_id"]),
            "prompt": str(row["prompt"]),
            "output": str(row["output"]),
        }
        for row in rows
        if row["split"] == "extraction"
    ]


def run(config_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R20 source output exists: {output}")
    root = Path(__file__).resolve().parents[2]
    load_config(root, config_path)
    output.mkdir(parents=True)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    cpu_started = time.process_time()
    load_started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    load_seconds = time.perf_counter() - load_started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    observations = []
    counters = {
        "raw_source_prompts": 0,
        "raw_source_prompt_utf8_bytes": 0,
        "rendered_prompt_utf8_bytes": 0,
        "rendered_prompt_token_instances": 0,
        "teacher_generated_tokens": 0,
        "teacher_output_bytes": 0,
    }
    inference_started = time.perf_counter()
    for split in ("extraction", "evaluation"):
        for row in public_rows(split):
            rendered = _render_chat(tokenizer, SYSTEM, str(row["prompt"]))
            input_tokens = tokenizer.encode(rendered, add_special_tokens=False)
            completion, output_tokens = _generate(tokenizer, model, rendered, MAX_NEW_TOKENS)
            completion = completion.strip().replace("\r\n", "\n")
            observations.append(
                {
                    **row,
                    "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "rendered_prompt_utf8_bytes": len(rendered.encode()),
                    "input_token_count": len(input_tokens),
                    "output": completion,
                    "output_sha256": hashlib.sha256(completion.encode()).hexdigest(),
                    "output_token_count": output_tokens,
                    "functional_exact": completion == row["expected"],
                }
            )
            counters["raw_source_prompts"] += 1
            counters["raw_source_prompt_utf8_bytes"] += len(str(row["prompt"]).encode())
            counters["rendered_prompt_utf8_bytes"] += len(rendered.encode())
            counters["rendered_prompt_token_instances"] += len(input_tokens)
            counters["teacher_generated_tokens"] += output_tokens
            counters["teacher_output_bytes"] += len(completion.encode())
    inference_seconds = time.perf_counter() - inference_started
    peak_gpu = int(torch.cuda.max_memory_allocated())
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "source_observations.jsonl"
    write_jsonl_once(rows_path, observations)
    records = _compiler_records(observations)
    try:
        _, diagnostics = infer_package(records)
        authorized = True
    except R20CompilerError as exc:
        diagnostics = {"error": str(exc)}
        authorized = False
    snapshot_files = [path for path in snapshot.rglob("*") if path.is_file()]
    memory = psutil.Process().memory_info()
    peak_cpu_rss = int(getattr(memory, "peak_wset", memory.rss))
    cpu_seconds = time.process_time() - cpu_started
    prompts = [str(row["prompt"]) for row in observations]
    teacher_outputs = [str(row["output"]) for row in observations]
    exact_by_task = {
        task: sum(
            row["functional_exact"]
            for row in observations
            if row["split"] == "evaluation" and row["task"] == task
        )
        for task in TASKS
    }
    receipt = {
        "format": "abi-r20-public-source-acquisition/1",
        "verdict": "PASS_SOURCE_COMPILABILITY" if authorized else "FAIL_SOURCE_COMPILABILITY",
        "compiler_authorized": authorized,
        "config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(Path(__file__).with_name("PUBLIC_PROTOCOL.md")),
        "source": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "snapshot_path_name": snapshot.name,
            "parameters": parameters,
            "training_steps": 0,
            "device": "cuda",
            "present_at_compilation": False,
            "present_at_package_execution": False,
        },
        "metrics": {
            "source_rows": len(observations),
            "extraction_rows": len(records),
            "evaluation_rows": len(observations) - len(records),
            "extraction_functional_exact": sum(
                row["functional_exact"] for row in observations if row["split"] == "extraction"
            ),
            "evaluation_functional_exact": sum(exact_by_task.values()),
            "evaluation_exact_by_task": exact_by_task,
            "compiler_diagnostics": diagnostics,
        },
        "artifacts": {"source_rows": {"path": rows_path.name, "sha256": sha256_file(rows_path)}},
        "information_accounting": {
            **counters,
            "unique_source_prompt_utf8_bytes": sum(len(value.encode()) for value in set(prompts)),
            "unique_teacher_output_utf8_bytes": sum(
                len(value.encode()) for value in set(teacher_outputs)
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "cpu_process_seconds": cpu_seconds,
            "cpu_process_hours": cpu_seconds / 3600,
            "source_load_seconds": load_seconds,
            "source_inference_seconds": inference_seconds,
            "source_model_inference_hours": inference_seconds / 3600,
            "peak_gpu_memory_bytes": peak_gpu,
            "peak_cpu_rss_bytes_observed": peak_cpu_rss,
            "gpu_name": gpu_name,
            "external_hardware_used": False,
            "source_parameters": parameters,
            "source_snapshot_files": len(snapshot_files),
            "source_snapshot_disk_bytes": sum(path.stat().st_size for path in snapshot_files),
            "source_observation_artifact_bytes": rows_path.stat().st_size,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "frozen_source_parameters_copied": 0,
            "final_imported_substrate_parameters": 0,
            "bridge_parameters_trained": 0,
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
