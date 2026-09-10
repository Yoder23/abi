"""Acquire teacher responses for the committed R21 hidden split."""

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
    sha256_file,
    write_json_once,
    write_jsonl_once,
)

from .acquire import OUTPUT_SYSTEM
from .hash_assurance_binding import selfless_evidence_hash
from .hidden_binding import load_hidden_config, load_seed_reveal
from .hidden_protocol import hidden_rows
from .protocol import score_output


def validate_hidden_source(
    root: Path, config_path: Path, reveal_path: Path, source_run: Path
) -> list[dict[str, Any]]:
    config = load_hidden_config(root, config_path)
    seed = load_seed_reveal(config, reveal_path)
    expected = hidden_rows(seed)
    receipt = json.loads((source_run / "receipt.json").read_text(encoding="utf-8"))
    rows_path = source_run / "source_observations.jsonl"
    try:
        rows = [
            json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise R14Error("R21 hidden source rows unreadable") from exc
    qualities = [
        score_output(
            reference,
            str(row.get("teacher_output", "")),
            str(row.get("teacher_output", "")),
        )
        for row, reference in zip(rows, expected)
    ]
    if (
        receipt.get("format") != "abi-r21-hidden-source/1"
        or receipt.get("verdict") != "PASS_HIDDEN_SOURCE_ACQUIRED"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or receipt.get("seed_reveal_sha256") != sha256_file(reveal_path)
        or receipt.get("evidence_sha256") != selfless_evidence_hash(receipt)
        or receipt.get("source_rows_sha256") != sha256_file(rows_path)
        or len(rows) != 120
        or [row.get("record_id") for row in rows] != [row["record_id"] for row in expected]
        or any(
            row.get("teacher_output_sha256")
            != hashlib.sha256(str(row.get("teacher_output", "")).encode()).hexdigest()
            for row in rows
        )
        or [row.get("teacher_quality") for row in rows] != qualities
        or receipt.get("teacher_functional") != sum(row["functional_pass"] for row in qualities)
        or receipt.get("teacher_non_hallucinating")
        != sum(row["hallucination_pass"] for row in qualities)
        or receipt.get("teacher_non_collapsed")
        != sum(not row["repetition_collapse"] for row in qualities)
    ):
        raise R14Error("R21 hidden source evidence failed")
    return rows


def run(config_path: Path, reveal_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise R14Error(f"immutable R21 hidden source exists: {output}")
    root = Path(__file__).resolve().parents[2]
    config = load_hidden_config(root, config_path)
    seed = load_seed_reveal(config, reveal_path)
    rows = hidden_rows(seed)
    if not torch.cuda.is_available():
        raise R14Error("R21 hidden source acquisition requires CUDA")
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
    input_tokens = 0
    output_tokens = 0
    output_bytes = 0
    inference_started = time.perf_counter()
    for index, row in enumerate(rows, 1):
        rendered = _render_chat(tokenizer, OUTPUT_SYSTEM, row["prompt"])
        response, tokens = _generate(
            tokenizer, model, rendered, int(config["source"]["max_new_tokens"])
        )
        response = response.strip().replace("\r\n", "\n")
        score = score_output(row, response, response)
        observations.append(
            {
                **row,
                "teacher_output": response,
                "teacher_output_sha256": hashlib.sha256(response.encode()).hexdigest(),
                "teacher_output_token_count": tokens,
                "teacher_quality": score,
                "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            }
        )
        input_tokens += len(tokenizer.encode(rendered, add_special_tokens=False))
        output_tokens += tokens
        output_bytes += len(response.encode())
        peak_rss = max(peak_rss, int(process.memory_info().rss))
        if index == 1 or index % 20 == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "hidden_source_rows": index,
                        "functional": sum(
                            item["teacher_quality"]["functional_pass"] for item in observations
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
    snapshot_files = [path for path in snapshot.rglob("*") if path.is_file()]
    receipt = {
        "format": "abi-r21-hidden-source/1",
        "verdict": "PASS_HIDDEN_SOURCE_ACQUIRED",
        "config_sha256": sha256_file(config_path),
        "seed_reveal_sha256": sha256_file(reveal_path),
        "source_rows_sha256": sha256_file(rows_path),
        "rows": len(observations),
        "teacher_functional": sum(
            row["teacher_quality"]["functional_pass"] for row in observations
        ),
        "teacher_non_hallucinating": sum(
            row["teacher_quality"]["hallucination_pass"] for row in observations
        ),
        "teacher_non_collapsed": sum(
            not row["teacher_quality"]["repetition_collapse"] for row in observations
        ),
        "source": {
            "model_id": config["source"]["model_id"],
            "revision": config["source"]["revision"],
            "parameters": parameters,
            "device": "cuda",
            "training_steps": 0,
            "present_at_package_execution": False,
        },
        "information_accounting": {
            "raw_prompts": len(rows),
            "unique_prompt_bytes": sum(len(row["prompt"].encode()) for row in rows),
            "rendered_prompt_token_instances": input_tokens,
            "teacher_generated_tokens": output_tokens,
            "teacher_output_bytes": output_bytes,
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
    }
    receipt["evidence_sha256"] = selfless_evidence_hash(receipt)
    write_json_once(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed-reveal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run(args.config, args.seed_reveal, args.output)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
