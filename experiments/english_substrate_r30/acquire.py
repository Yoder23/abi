"""Acquire free labels and outputs from the frozen R30 source teacher."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import psutil
import torch

from experiments.factual_semantic_r16.public_qualification import _generate, _load_source, _render_chat
from experiments.foreign_capability_r14.core import evidence_hash, write_json_once, write_jsonl_once
from .protocol import normalize_free_label, rows


MODEL_ID = "Qwen/Qwen2-7B-Instruct"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
OUTPUT_SYSTEM = (
    "Perform the requested English-language operation using only the SUPPLIED MATERIAL. "
    "Preserve its unique identifier and relevant numbers. Never add real-world facts. "
    "If the material says an answer is unsupported, explicitly abstain. Return only the answer."
)
LABEL_SYSTEM = (
    "Name the primary language capability requested by the user. Invent a short descriptive "
    "lowercase label of one to three words. No label choices are supplied. Return only the label."
)


def run(output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"immutable R30 acquisition exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("R30 source acquisition requires the declared CUDA device")
    output.mkdir(parents=True)
    source_rows = rows("train") + rows("evaluation")
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer, model, snapshot = _load_source(MODEL_ID, REVISION)
    loaded = time.perf_counter() - started
    observations = []
    generated_tokens = 0
    rendered_tokens = 0
    for index, row in enumerate(source_rows, 1):
        rendered = _render_chat(tokenizer, OUTPUT_SYSTEM, row["prompt"])
        answer, count = _generate(tokenizer, model, rendered, 144)
        answer = answer.strip().replace("\r\n", "\n")
        label_raw = ""
        label_tokens = 0
        if row["split"] == "train":
            label_prompt = _render_chat(tokenizer, LABEL_SYSTEM, row["prompt"])
            label_raw, label_tokens = _generate(tokenizer, model, label_prompt, 12)
            label_raw = label_raw.strip()
            rendered_tokens += len(tokenizer.encode(label_prompt, add_special_tokens=False))
        canonical = normalize_free_label(label_raw) if label_raw else None
        observations.append(
            {
                **row,
                "teacher_output": answer,
                "teacher_output_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "teacher_output_tokens": count,
                "free_label": label_raw,
                "free_label_tokens": label_tokens,
                "normalized_label": canonical,
                "label_exact": canonical == row["oracle_task"] if canonical else False,
            }
        )
        generated_tokens += count + label_tokens
        rendered_tokens += len(tokenizer.encode(rendered, add_special_tokens=False))
        peak_rss = max(peak_rss, process.memory_info().rss)
        if index == 1 or index % 24 == 0 or index == len(source_rows):
            train_seen = [item for item in observations if item["split"] == "train"]
            print(json.dumps({"rows": index, "labels_exact": sum(item["label_exact"] for item in train_seen), "labels_seen": len(train_seen), "seconds": time.perf_counter() - started}), flush=True)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    rows_path = output / "source_rows.jsonl"
    write_jsonl_once(rows_path, observations)
    train = [row for row in observations if row["split"] == "train"]
    counts = {task: {"rows": 0, "normalized": 0, "exact": 0} for task in sorted({row["oracle_task"] for row in train})}
    for row in train:
        item = counts[row["oracle_task"]]
        item["rows"] += 1
        item["normalized"] += int(row["normalized_label"] is not None)
        item["exact"] += int(row["label_exact"])
    exact = sum(row["label_exact"] for row in train)
    result = {
        "format": "abi-r30-public-source/1",
        "verdict": "PASS_SOURCE" if exact >= 276 and all(item["exact"] >= 22 for item in counts.values()) else "FAIL_SOURCE",
        "source": {"model_id": MODEL_ID, "revision": REVISION, "snapshot": str(snapshot), "teacher_training_steps": 0},
        "metrics": {"rows": len(observations), "train_rows": len(train), "evaluation_rows": len(observations) - len(train), "label_exact": exact, "label_by_task": counts},
        "information_accounting": {
            "raw_source_prompts": len(observations) + len(train),
            "unique_raw_prompt_bytes": sum(len(row["prompt"].encode()) for row in observations),
            "teacher_output_bytes": sum(len(row["teacher_output"].encode()) for row in observations),
            "teacher_generated_tokens": generated_tokens,
            "rendered_prompt_tokens": rendered_tokens,
            "logits_stored": 0,
            "hidden_activations_stored": 0,
            "source_parameters_copied": 0,
            "source_load_seconds": loaded,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cpu_rss_bytes": int(peak_rss),
            "external_hardware_used": False,
        },
        "artifacts": {"source_rows": {"path": rows_path.name, "sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest()}},
        "claim_ceiling": "PUBLIC_FALSIFICATION_ONLY_NOT_GENERAL_ENGLISH",
        "full_abi_moonshot": "OPEN",
    }
    result["evidence_sha256"] = evidence_hash(result)
    write_json_once(output / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
